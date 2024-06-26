import sqlite3
import requests
import json
from datetime import datetime
import bittensor as bt


def fetch_predictions_from_db(db_path):
    """
    Fetch predictions from the SQLite3 database.

    :param db_path: Path to the SQLite3 database file
    :return: List of dictionaries containing prediction data
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    cursor = conn.cursor()

    # First, let's check the available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Available tables:", [table[0] for table in tables])

    # The 'predictions' table exists, so let's check its structure
    cursor.execute("PRAGMA table_info(predictions)")
    columns = cursor.fetchall()
    print("Columns in 'predictions' table:", [col[1] for col in columns])

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
    print("Executing query:", query)

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        predictions = []
        for row in rows:
            predictions.append(dict(row))

        print(f"Fetched {len(predictions)} predictions")
        return predictions
    except sqlite3.OperationalError as e:
        print(f"Error executing query: {e}")
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
    :return: Tuple of (status_code, API response content)
    """
    url = "https://www.bettensor.com/API/Predictions/"

    transformed_data = []

    for prediction in predictions:
        transformed_prediction = {
            "externalGameId": prediction["teamGameID"],
            "minerHotKey": prediction["minerId"],
            "minerColdKey": "dummy_coldkey",
            "predictionDate": prediction["predictionDate"],
            "predictedOutcome": 0,
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
    predictions = fetch_predictions_from_db(db_path)
    if predictions:
        bt.logging.debug("Sending predictions to the Bettensor website.")
        return send_predictions(predictions, db_path)
    else:
        print("No predictions found in the database.")
        return None
