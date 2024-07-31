import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.serving import run_simple
from bettensor.utils.database_manager import get_db_manager
import jwt
from functools import wraps
import sqlite3

app = Flask(__name__)

# Configuration
class Config:
    LOCAL_INTERFACE = os.environ.get('LOCAL_INTERFACE', 'True').lower() == 'true'
    CENTRAL_SERVER = os.environ.get('CENTRAL_SERVER', 'False').lower() == 'true'
    JWT_SECRET = 'your-secret-key'  # Change this in production and store securely

config = Config()

# Apply CORS only if using local interface
if config.LOCAL_INTERFACE:
    CORS(app)

# JWT token verification
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    return decorated

def get_miner_uids():
    miner_uids = []
    with open("data/miner_env.txt", "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            uid = parts[0].split("=")[1]
            miner_uids.append(uid)
    return miner_uids

@app.route('/submit_prediction', methods=['POST'])
@token_required
def submit_prediction():
    data = request.json
    miner_uid = data.get('miner_uid')
    if not miner_uid:
        return jsonify({'message': 'Miner UID is required'}), 400

    db_manager = get_db_manager(miner_uid)
    
    with db_manager.get_cursor() as cursor:
        cursor.execute(
            """INSERT INTO predictions (predictionID, teamGameID, minerID, predictionDate, teamA, teamB, teamAodds, teamBodds, tieOdds, predictedOutcome, wager, outcome, canOverwrite) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data['predictionID'],
                data['teamGameID'],
                data['minerID'],
                data['predictionDate'],
                data['teamA'],
                data['teamB'],
                data['teamAodds'],
                data['teamBodds'],
                data['tieOdds'],
                data['predictedOutcome'],
                data['wager'],
                data['outcome'],
                data['canOverwrite'],
            ),
        )
        cursor.connection.commit()
    
    return jsonify({'message': 'Prediction submitted successfully'}), 200

""" @app.route('/get_games', methods=['GET'])
def get_games():
    if not config.LOCAL_INTERFACE:
        return jsonify({'message': 'Endpoint not available'}), 403
    
    miner_uid = request.args.get('miner_uid')
    if not miner_uid:
        return jsonify({'message': 'Miner UID is required'}), 400

    db_manager = get_db_manager(miner_uid)
    games = {}
    
    with db_manager.get_cursor() as cursor:
        cursor.execute("SELECT * FROM games WHERE active = 0")
        columns = [column[0] for column in cursor.description]
        for row in cursor.fetchall():
            games[row[0]] = {columns[i]: row[i] for i in range(len(columns))}
    
    return jsonify(games), 200 """

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    if not config.LOCAL_INTERFACE:
        return jsonify({'message': 'Endpoint not available'}), 403
    
    miner_uid = request.args.get('miner_uid')
    if not miner_uid:
        return jsonify({'message': 'Miner UID is required'}), 400

    db_manager = get_db_manager(miner_uid)
    predictions = {}
    
    with db_manager.get_cursor() as cursor:
        cursor.execute("SELECT * FROM predictions")
        columns = [column[0] for column in cursor.description]
        for row in cursor.fetchall():
            predictions[row[0]] = {columns[i]: row[i] for i in range(len(columns))}
    
    return jsonify(predictions), 200

@app.route('/get_miners', methods=['GET'])
def get_miners():
    '''
    called from website - returns a list of miner uids for the coldkey
    '''
    miner_uids = get_miner_uids()
    return jsonify(miner_uids), 200

@app.route('/submit_prediction', methods=['POST'])
@token_required
def submit_prediction():
    '''
    called from website - submits a prediction to the miner, which is then submitted to validators
    '''
    data = request.json
    miner_uid = data.get('miner_uid')
    if not miner_uid:
        return jsonify({'message': 'Miner UID is required'}), 400

@app.route('/heartbeat', methods=['GET'])
@token_required
def heartbeat():
    '''
    called every 15 seconds - returns the most recent miner stats
    
    '''
    return jsonify({'message': 'Miner is alive'}), 200




def start_server():
    if config.LOCAL_INTERFACE:
        # Run the server locally, only accessible from localhost
        run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
    elif config.CENTRAL_SERVER:
        # Run the server with specific host and port for central server access
        # Make sure to set up proper firewall rules and SSL in production
        app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
    else:
        print("No valid server configuration found. Please set either LOCAL_INTERFACE or CENTRAL_SERVER to True.")

if __name__ == '__main__':
    start_server()