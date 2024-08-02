import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.serving import run_simple
import jwt
from functools import wraps
from bettensor.miner.database.database_manager import get_db_manager
import redis
import uuid
import bittensor as bt
import argparse

app = Flask(__name__)

# Configuration
class Config:
    LOCAL_SERVER = os.environ.get('LOCAL_SERVER', 'False').lower() == 'true'
    CENTRAL_SERVER = os.environ.get('CENTRAL_SERVER', 'True').lower() == 'true'
    JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key')  # Change this in production and store securely
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

config = Config()

# Apply CORS only if using local server
if config.LOCAL_SERVER:
    CORS(app)

# Initialize Redis client
redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

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

    # Generate a unique message ID
    message_id = str(uuid.uuid4())
    data['message_id'] = message_id

    # Publish the prediction data to the miner's channel
    redis_client.publish(f'miner:{miner_uid}', json.dumps(data))

    # Wait for the response (with a timeout)
    response = redis_client.blpop(f'response:{message_id}', timeout=30)
    if response:
        _, result = response
        result = json.loads(result)
        return jsonify(result), 200
    else:
        return jsonify({'message': 'Prediction submission timed out'}), 408

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    if not config.LOCAL_SERVER:
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
    miner_uids = get_miner_uids()
    return jsonify(miner_uids), 200

@app.route('/heartbeat', methods=['GET'])
@token_required
def heartbeat():
    return jsonify({'message': 'Miner interface server is alive'}), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bettensor Miner Interface Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    if config.LOCAL_SERVER:
        app.run(host=args.host, port=args.port)
    elif config.CENTRAL_SERVER:
        app.run(host=args.host, port=args.port, ssl_context='adhoc')
    else:
        print("No valid server configuration found. Please set either LOCAL_SERVER or CENTRAL_SERVER to True.")