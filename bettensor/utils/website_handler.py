import sqlite3
import requests
import json
from datetime import datetime
import bittensor as bt
import asyncio

async def create_keys_table(db_path: str):
    """
    Creates keys table in db
    """
    conn = await asyncio.to_thread(sqlite3.connect, db_path)
    cursor = await asyncio.to_thread(conn.cursor)
    await asyncio.to_thread(cursor.execute, """
        CREATE TABLE IF NOT EXISTS keys (
            hotkey TEXT PRIMARY KEY,
            coldkey TEXT
        )
    """)
    await asyncio.to_thread(conn.commit)
    await asyncio.to_thread(conn.close)

async def get_or_update_coldkey(db_path: str, hotkey: str) -> str:
    """
    Retrieves coldkey from metagraph if it doesnt exist in keys table
    """
    conn = await asyncio.to_thread(sqlite3.connect, db_path)
    cursor = await asyncio.to_thread(conn.cursor)

    # Check if the hotkey exists in the keys table
    result = await asyncio.to_thread(cursor.execute, "SELECT coldkey FROM keys WHERE hotkey = ?", (hotkey,))
    result = await asyncio.to_thread(result.fetchone)

    if result:
        await asyncio.to_thread(conn.close)
        return result[0]
    else:
        # If not found, fetch from Bittensor and insert into the database
        metagraph = bt.metagraph(netuid=30)
        await asyncio.to_thread(metagraph.sync)  # Sync with the network to get the latest data

        for neuron in metagraph.neurons:
            if neuron.hotkey == hotkey:
                coldkey = neuron.coldkey
                await asyncio.to_thread(cursor.execute, "INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)", (hotkey, coldkey))
                await asyncio.to_thread(conn.commit)
                await asyncio.to_thread(conn.close)
                return coldkey

        # If coldkey is not found, insert "dummy_coldkey"
        await asyncio.to_thread(cursor.execute, "INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)", (hotkey, "dummy_coldkey"))
        await asyncio.to_thread(conn.commit)
        await asyncio.to_thread(conn.close)
        return "dummy_coldkey"

async def fetch_predictions_from_db(db_path):
    """
    Fetch predictions from the SQLite3 database.

    :param db_path: Path to the SQLite3 database file
    :return: List of dictionaries containing prediction data
    """
    conn = await asyncio.to_thread(sqlite3.connect, db_path)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    cursor = await asyncio.to_thread(conn.cursor)

    tables = await asyncio.to_thread(cursor.execute, "SELECT name FROM sqlite_master WHERE type='table';")
    tables = await asyncio.to_thread(tables.fetchall)

    columns = await asyncio.to_thread(cursor.execute, "PRAGMA table_info(predictions)")
    columns = await asyncio.to_thread(columns.fetchall)

    # Construct the query based on the actual columns in the table
    available_columns = [col[1] for col in columns]
    query_columns = [
        col
        for col in [
            "predictionID",
            "teamGameID",
            "minerId",
            "predictionDate",
            "predictedOutcome",
            "teamA",
            "teamB",
            "wager",
            "teamAodds",
            "teamBodds",
            "tieOdds",
            "canOverwrite",
            "outcome",
            "sent_to_site"
        ]
        if col in available_columns
    ]

    query = f"SELECT {', '.join(query_columns)} FROM predictions WHERE sent_to_site = 0"

    try:
        rows = await asyncio.to_thread(cursor.execute, query)
        rows = await asyncio.to_thread(rows.fetchall)

        predictions = [dict(row) for row in rows]
        return predictions
    except sqlite3.OperationalError as e:
        bt.logging.trace(e)
        return []
    finally:
        await asyncio.to_thread(conn.close)

async def update_sent_status(db_path, prediction_ids):
    conn = await asyncio.to_thread(sqlite3.connect, db_path)
    cursor = await asyncio.to_thread(conn.cursor)
    
    try:
        await asyncio.to_thread(cursor.executemany,
            "UPDATE predictions SET sent_to_site = 1 WHERE predictionID = ?",
            [(pid,) for pid in prediction_ids]
        )
        await asyncio.to_thread(conn.commit)
        print(f"Updated sent_to_site status for {len(prediction_ids)} predictions")
    except sqlite3.Error as e:
        print(f"Error updating sent_to_site status: {e}")
    finally:
        await asyncio.to_thread(conn.close)

async def send_predictions(predictions, db_path):
    """
    Send predictions to the Bettensor API.

    :param predictions: List of dictionaries containing prediction data
    :param db_path: Path to the SQLite3 database file
    :return: Tuple of (status_code, API response content)
    """
    url = "https://www.bettensor.com/API/Predictions/"

    transformed_data = []

    for prediction in predictions:
        hotkey = prediction["minerId"]
        try:
            coldkey = await get_or_update_coldkey(db_path, hotkey)
        except ValueError as e:
            bt.logging.error(e)
            coldkey = "dummy_coldkey"

        transformed_prediction = {
            "externalGameId": prediction["teamGameID"],
            "minerHotKey": hotkey,
            "minerColdKey": coldkey,
            "predictionDate": prediction["predictionDate"],
            "predictedOutcome": prediction["predictedOutcome"],
            "wager": prediction["wager"],
        }

        try:
            date = datetime.strptime(prediction["predictionDate"], "%Y-%m-%d %H:%M:%S")
            transformed_prediction["predictionDate"] = date.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        except ValueError:
            pass

        transformed_data.append(transformed_prediction)

    headers = {"Content-Type": "application/json"}

    bt.logging.info(f"Sending {len(transformed_data)} predictions to API")
    bt.logging.debug(
        f"First prediction (for debugging): {json.dumps(transformed_data[0], indent=2)}"
    )
    print(transformed_data)
    try:
        response = await asyncio.to_thread(requests.post,
            url, data=json.dumps(transformed_data), headers=headers
        )
        if response.status_code == 200 or response.status_code == 201:
            await update_sent_status(db_path, [p['predictionID'] for p in predictions])
        bt.logging.info(f"Response status code: {response.status_code}")
        bt.logging.debug(f"Response content: {response.text}")
        return response.status_code

    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Error sending predictions: {e}")
        return None, str(e)

async def fetch_and_send_predictions(db_path):
    """
    Fetch predictions from the database and send them to the API.

    :param db_path: Path to the SQLite3 database file
    :return: API response
    """
    await create_keys_table(db_path)  # Ensure the keys table exists
    predictions = await fetch_predictions_from_db(db_path)
    if predictions:
        bt.logging.debug("Sending predictions to the Bettensor website.")
        return await send_predictions(predictions, db_path)
    else:
        print("No predictions found in the database.")
        return None