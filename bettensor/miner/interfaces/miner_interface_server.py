import os
import json
import sys
import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import requests
from werkzeug.serving import run_simple
import jwt as pyjwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from bettensor.protocol import TeamGamePrediction
from functools import wraps
from bettensor.miner.database.database_manager import get_db_manager
import uuid
import bittensor as bt
import argparse
from bettensor.miner.interfaces.redis_interface import RedisInterface
import psycopg2
from psycopg2.extras import RealDictCursor
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

# Set up logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

app = Flask(__name__)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

connected_miners = []
server_start_time = time.time()

CENTRAL_SERVER_IP = "your_central_server_ip_here"  # Replace with actual IP
MAX_REQUESTS = 100
TIME_WINDOW = 60  # seconds
BLACKLIST_DURATION = 3600  # 1 hour

blacklist = {}
request_count = {}

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10 per minute"]
)

# Configuration
class Config:
    LOCAL_SERVER = os.environ.get('LOCAL_SERVER', 'False').lower() == 'true'
    CENTRAL_SERVER = os.environ.get('CENTRAL_SERVER', 'True').lower() == 'true'
    JWT_SECRET = os.environ.get('JWT_SECRET', 'bettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensor')  # Change this in production and store securely
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

config = Config()

print(f"JWT_SECRET: {config.JWT_SECRET}")

# Apply CORS only if using local server
if config.LOCAL_SERVER:
    CORS(app)

# Initialize Redis client
redis_interface = RedisInterface(host=config.REDIS_HOST, port=config.REDIS_PORT)
if not redis_interface.connect():
    bt.logging.error("Failed to connect to Redis. Exiting.")
    sys.exit(1)

# JWT token verification
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Remove 'Bearer ' prefix
            except IndexError:
                token = auth_header  # No 'Bearer ' prefix

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            # Decode the token without verifying the audience
            pyjwt.decode(token, config.JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        except ExpiredSignatureError:
            bt.logging.error("Token has expired")
            return jsonify({'message': 'Token has expired!'}), 401
        except InvalidTokenError as e:
            bt.logging.error(f"Invalid token: {str(e)}")
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(*args, **kwargs)

    return decorated

def get_miner_uids():
    pass

@app.route('/submit_predictions', methods=['POST'])
@limiter.limit("50 per minute")
@token_required
def submit_predictions():
    data = request.json
    miner_hotkey = data.get('minerID')
    predictions = data.get('predictions', [])

    bt.logging.info(f"Received prediction submission for miner: {miner_hotkey}")

    # Get miner uid from hotkey
    miner_uid = get_miner_uid(miner_hotkey)
    bt.logging.info(f"Miner UID: {miner_uid}")

    # Generate a unique message ID 
    message_id = str(uuid.uuid4())
    bt.logging.info(f"Generated message ID: {message_id}")

    # Prepare the message to be published
    message = {
        'message_id': message_id,
        'minerID': miner_hotkey,
        'predictions': predictions
    }

    # Publish the prediction data to the miner's channel
    channel = f'miner:{miner_uid}:{miner_hotkey}'
    bt.logging.info(f"Publishing to channel: {channel}")
    if not redis_interface.publish(channel, json.dumps(message)):
        bt.logging.error("Failed to publish prediction request to Redis")
        return jsonify({'message': 'Failed to publish prediction request'}), 500

    bt.logging.info("Waiting for response...")

    # Wait for the response with a timeout
    start_time = time.time()
    timeout = 60  # 60 seconds timeout
    while time.time() - start_time < timeout:
        response = redis_interface.get(f'response:{message_id}')
        if response:
            bt.logging.info("Response received")
            result = json.loads(response)
            return jsonify(result), 200
        time.sleep(0.1)  # Short sleep to prevent busy-waiting

    bt.logging.error("Prediction submission timed out")
    return jsonify({'message': 'Prediction submission timed out after 60 seconds'}), 408

@app.route('/get_predictions', methods=['GET'])
@limiter.limit("50 per minute")
@token_required
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

def get_miners() -> list:
    bt.logging.debug("Attempting to retrieve miners from database")
    try:
        conn = psycopg2.connect(
            dbname="bettensor",
            user="root",
            password="bettensor_password",
            host="localhost"
        )
        bt.logging.debug("Database connection established")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    miner_hotkey,
                    miner_uid,
                    miner_rank,
                    miner_cash,
                    miner_current_incentive,
                    miner_last_prediction_date,
                    miner_lifetime_earnings,
                    miner_lifetime_wager,
                    miner_lifetime_predictions,
                    miner_lifetime_wins,
                    miner_lifetime_losses,
                    miner_win_loss_ratio,
                    last_daily_reset
                FROM miner_stats
            """)
            miners = cur.fetchall()
        conn.close()
        bt.logging.debug(f"Retrieved {len(miners)} miners from database")
        return [dict(miner) for miner in miners]
    except psycopg2.Error as e:
        bt.logging.error(f"Database error: {e}")
        return []
    except Exception as e:
        bt.logging.error(f"Unexpected error in get_miners: {str(e)}")
        return []

def get_miner_uid(hotkey: str):
    #use postgres to get the miner uid from the hotkey
    with psycopg2.connect(
            dbname="bettensor",
            user="root",
            password="bettensor_password",
            host="localhost"
        ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM miner_stats WHERE miner_hotkey = %s", (hotkey,))
            miner = cur.fetchone()
        return miner['miner_uid']

@app.route('/heartbeat', methods=['GET'])
@limiter.limit("2 per minute")
@token_required
def heartbeat():
    '''
    When connected to the central server, this endpoint will be called every minute.
    It is used to check if the miner(s) are still connected to the central server.
    If the miner(s) are not connected, the central server will assume the miner(s) are offline and will not send any more requests to the miner(s).
    schema:
        "headers": {
            "Authorization": JWT for verifying the central server requests
        }
        data: {
            
        }

    returns:
    {
        'http status': '200' or '401' or '403' or '500',
        'data': {
            'tokenStatus': 'valid' or 'invalid',
            'miners': [MinerStats]
        }
    }

    '''
    # Update server information in Redis
    update_server_info()
    token = request.headers.get('Authorization', '').split(" ")[-1]  # Get token from Authorization header
    token_status = get_token_status(token)
    response = {
        "tokenStatus": token_status,
        "miners": get_miners()
    }
    return jsonify(response), 200


def get_token_status(token: str):
    with open("token_store.json", "r") as f:
        token_data = json.load(f)
        if token_data["jwt"] == token and token_data["revoked"] == False:
            return "valid"
        else:
            return "invalid"

def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def update_server_info():
    bt.logging.debug("Updating server info")
    uptime = int(time.time() - server_start_time)
    try:
        ip = requests.get('https://api.ipify.org').text
        bt.logging.debug(f"Retrieved IP: {ip}")
    except Exception as e:
        ip = "Unable to determine IP"
        bt.logging.error(f"Error retrieving IP: {str(e)}")
    
    server_info = {
        "status": "running",
        "ip": ip,
        "port": str(SERVER_PORT),
        "connected_miners": str(len(get_miners())),
        "uptime": f"{uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s"
    }
    
    bt.logging.debug(f"Server info to update: {server_info}")
    
    # Update each field individually
    for key, value in server_info.items():
        try:
            result = redis_interface.set(f"server_info:{key}", value)
            bt.logging.debug(f"Set {key}: {value} - Result: {result}")
        except Exception as e:
            bt.logging.error(f"Error setting {key}: {str(e)}")

    bt.logging.debug("Server info update complete")

    # Verify the data was set correctly
    for key in server_info.keys():
        value = redis_interface.get(f"server_info:{key}")
        bt.logging.debug(f"Verified {key}: {value}")

def handle_token_management():
    r = get_redis_client()
    p = r.pubsub()
    p.subscribe("token_management")

    for message in p.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            action = data["action"]
            token_data = data["data"]

            if action == "revoke":
                with open("token_store.json", "w") as f:
                    token_data["revoked"] = True
                    json.dump(token_data, f)
                r.rpush("token_management_response", json.dumps({"message": "Token revoked successfully."}))
            elif action == "check":
                stored_token = json.loads(open("token_store.json", "r").read())
                if stored_token["jwt"] == token_data["jwt"] and stored_token["signature"] == token_data["signature"]:
                    if stored_token["revoked"]:
                        r.rpush("token_management_response", json.dumps({"message": "Token is revoked."}))
                    else:
                        r.rpush("token_management_response", json.dumps({"message": "Token is valid and active."}))
                else:
                    r.rpush("token_management_response", json.dumps({"message": "Invalid token."}))

# Start the token management handler in a separate thread
token_management_thread = threading.Thread(target=handle_token_management)
token_management_thread.start()

def update_server_info_periodically():
    bt.logging.info("Server info update thread started")
    while True:
        bt.logging.debug("Periodic update of server info")
        try:
            update_server_info()
        except Exception as e:
            bt.logging.error(f"Error updating server info: {str(e)}")
        bt.logging.debug("Sleeping for 60 seconds")
        time.sleep(60)  # Update every minute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bettensor Miner Interface Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    SERVER_PORT = args.port

    # Initialize and connect to Redis
    redis_interface = RedisInterface(host=config.REDIS_HOST, port=config.REDIS_PORT)
    if not redis_interface.connect():
        bt.logging.error("Failed to connect to Redis. Exiting.")
        sys.exit(1)

    # Update server info immediately
    bt.logging.info("Updating server info immediately")
    update_server_info()

    # Start a thread to update server info periodically
    server_info_thread = threading.Thread(target=update_server_info_periodically)
    server_info_thread.daemon = True
    server_info_thread.start()
    bt.logging.info("Server info update thread started")

    # Verify Redis connection
    try:
        redis_interface.ping()
        bt.logging.info("Redis connection verified")
    except Exception as e:
        bt.logging.error(f"Redis connection failed: {str(e)}")

    bt.logging.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True, threaded=True)



@app.before_request
def check_blacklist():
    ip = request.remote_addr
    if ip in blacklist:
        if time.time() - blacklist[ip] < BLACKLIST_DURATION:
            return jsonify({"error": "IP blacklisted"}), 403
        else:
            del blacklist[ip]
    
    if ip != CENTRAL_SERVER_IP:
        current_time = time.time()
        request_count[ip] = request_count.get(ip, [])
        request_count[ip] = [t for t in request_count[ip] if current_time - t < TIME_WINDOW]
        request_count[ip].append(current_time)
        
        if len(request_count[ip]) > MAX_REQUESTS:
            blacklist[ip] = current_time
            return jsonify({"error": "Too many requests, IP blacklisted"}), 429

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded"}), 429