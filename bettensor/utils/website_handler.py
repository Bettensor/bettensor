import sqlite3
import requests
import json
from datetime import datetime
import bittensor as bt

def create_keys_table(db_path: str):
    """
    Creates keys table in db
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS keys (
            hotkey TEXT PRIMARY KEY,
            coldkey TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_or_update_coldkey(db_path: str, hotkey: str) -> str:
    """
    Retrieves coldkey from metagraph if it doesnt exist in keys table
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the hotkey exists in the keys table
    cursor.execute("SELECT coldkey FROM keys WHERE hotkey = ?", (hotkey,))
    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        # If not found, fetch from Bittensor and insert into the database
        metagraph = bt.metagraph(netuid=30)
        metagraph.sync()  # Sync with the network to get the latest data

        for neuron in metagraph.neurons:
            if neuron.hotkey == hotkey:
                coldkey = neuron.coldkey
                cursor.execute("INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)", (hotkey, coldkey))
                conn.commit()
                conn.close()
                return coldkey

        # If coldkey is not found, insert "dummy_coldkey"
        cursor.execute("INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)", (hotkey, "dummy_coldkey"))
        conn.commit()
        conn.close()
        return "dummy_coldkey"

def fetch_predictions_from_db(db_path):
    """
    Fetch predictions from the SQLite3 database.

    :param db_path: Path to the SQLite3 database file
    :return: List of dictionaries containing prediction data
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    cursor.execute("PRAGMA table_info(predictions)")
    columns = cursor.fetchall()

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
        cursor.execute(query)
        rows = cursor.fetchall()

        predictions = []
        for row in rows:
            predictions.append(dict(row))

        return predictions
    except sqlite3.OperationalError as e:
        bt.logging.trace(e)
        return []
    finally:
        conn.close()

def update_sent_status(db_path, prediction_ids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.executemany(
            "UPDATE predictions SET sent_to_site = 1 WHERE predictionID = ?",
            [(pid,) for pid in prediction_ids]
        )
        conn.commit()
        print(f"Updated sent_to_site status for {len(prediction_ids)} predictions")
    except sqlite3.Error as e:
        print(f"Error updating sent_to_site status: {e}")
    finally:
        conn.close()    

def send_predictions(predictions, db_path):
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
            coldkey = get_or_update_coldkey(db_path, hotkey)
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
        response = requests.post(
            url, data=json.dumps(transformed_data), headers=headers
        )
        if response.status_code == 200 or response.status_code == 201:
            update_sent_status(db_path, [p['predictionID'] for p in predictions])
        bt.logging.info(f"Response status code: {response.status_code}")
        bt.logging.debug(f"Response content: {response.text}")
        return response.status_code

    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Error sending predictions: {e}")
        return None, str(e)

def fetch_and_send_predictions(db_path):
    """
    Fetch predictions from the database and send them to the API.

    :param db_path: Path to the SQLite3 database file
    :return: API response
    """
    create_keys_table(db_path)  # Ensure the keys table exists
    predictions = fetch_predictions_from_db(db_path)
    if predictions:
        bt.logging.debug("Sending predictions to the Bettensor website.")
        return send_predictions(predictions, db_path)
    else:
        print("No predictions found in the database.")
        return None
