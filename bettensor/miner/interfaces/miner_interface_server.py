import os
import json
import sys
import threading
import time
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import redis
import requests
from werkzeug.serving import run_simple
import jwt as pyjwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError, DecodeError
from bettensor.protocol import TeamGamePrediction
from functools import wraps
from bettensor.miner.database.database_manager import DatabaseManager
import uuid
import bittensor as bt
import argparse
from bettensor.miner.interfaces.redis_interface import RedisInterface
import psycopg2
from psycopg2.extras import RealDictCursor
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import traceback
from bettensor.miner.database.games import GamesHandler
import socket

# Set up logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

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
    JWT_SECRET = os.environ.get('JWT_SECRET', 'bettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensor')  # Change this in production and store securely
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

config = Config()

print(f"JWT_SECRET: {config.JWT_SECRET}")

# Apply CORS only if using local server

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
        bt.logging.info("Entering token_required decorator")
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Remove 'Bearer ' prefix
            except IndexError:
                token = auth_header  # No 'Bearer ' prefix

        bt.logging.info(f"Received token: {token[:10]}...") if token else bt.logging.info("No token received")

        if not token:
            bt.logging.warning("Token is missing")
            return jsonify({'message': 'Token is missing!'}), 401

        # Check if the token matches the one in token_store.json
        if os.path.exists("token_store.json"):
            try:
                with open("token_store.json", "r") as file:
                    stored_data = json.load(file)
                    stored_jwt = stored_data.get("jwt")
                    if stored_jwt and stored_jwt != token:
                        bt.logging.warning("Received token does not match the stored token")
                        return jsonify({'message': 'Invalid token!'}), 401
                    else:
                        bt.logging.info("Token matches stored token")
            except json.JSONDecodeError as e:
                bt.logging.error(f"Error decoding token_store.json: {str(e)}")
                return jsonify({'message': 'Error reading stored token!'}), 500
        else:
            bt.logging.warning("token_store.json not found")

        try:
            # Decode the token without verifying the signature
            decoded = pyjwt.decode(token, options={"verify_signature": False})
            bt.logging.info(f"Decoded token: {decoded}")
            
            # Now verify the signature separately
            pyjwt.decode(token, config.JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
            bt.logging.info("Token signature successfully verified")
        except ExpiredSignatureError:
            bt.logging.error("Token has expired")
            return jsonify({'message': 'Token has expired!'}), 401
        except (InvalidTokenError, pyjwt.DecodeError) as e:
            bt.logging.error(f"Invalid token: {str(e)}")
            bt.logging.error(f"JWT_SECRET used: {config.JWT_SECRET}")
            return jsonify({'message': 'Token is invalid!'}), 401

        bt.logging.info("Token validation successful")
        return f(*args, **kwargs)

    return decorated

def get_miner_uids():
    pass

@app.route('/submit_predictions', methods=['POST'])
@token_required
def submit_predictions():
    data = request.json
    app.logger.info(f"Entering submit_predictions endpoint")
    app.logger.info(f"Received data: {data}")

    miner_id = data.get('minerID')
    predictions = data.get('predictions', [])

    if not miner_id or not predictions:
        return jsonify({'error': 'Invalid request data'}), 400

    app.logger.info(f"Received prediction submission for miner: {miner_id}")
    app.logger.info(f"Number of predictions: {len(predictions)}")

    # Get miner UID
    miner_uid = get_miner_uid(miner_id)
    app.logger.info(f"Miner UID: {miner_uid}")

    if miner_uid is None:
        return jsonify({'error': 'Miner not found'}), 404

    # Generate a unique message ID
    message_id = str(uuid.uuid4())
    app.logger.info(f"Generated message ID: {message_id}")

    # Prepare the message with the correct action
    message = {
        'action': 'make_prediction',  # Set the action explicitly
        'message_id': message_id,
        'miner_id': miner_id,
        'predictions': predictions
    }

    # Publish the message to Redis
    channel = f'miner:{miner_uid}:{miner_id}'
    app.logger.info(f"Publishing to channel: {channel}")
    redis_client.publish(channel, json.dumps(message))

    # Wait for the response
    app.logger.info("Waiting for response...")
    response = redis_client.get(f'response:{message_id}')

    if response:
        return jsonify(json.loads(response))
    else:
        return jsonify({'error': 'No response from miner'}), 500

def get_miner_uid(hotkey: str):
    bt.logging.info(f"Getting miner UID for hotkey: {hotkey}")
    try:
        with psycopg2.connect(
                dbname="bettensor",
                user="root",
                password="bettensor_password",
                host="localhost"
            ) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM miner_stats WHERE miner_hotkey = %s", (hotkey,))
                miner = cur.fetchone()
            if miner:
                bt.logging.info(f"Found miner UID: {miner['miner_uid']}")
                return miner['miner_uid']
            else:
                bt.logging.warning(f"No miner found for hotkey: {hotkey}")
                return None
    except Exception as e:
        bt.logging.error(f"Error in get_miner_uid: {str(e)}")
        bt.logging.error(traceback.format_exc())
        raise

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

@app.route('/heartbeat')
def heartbeat():
    uptime = int(time.time() - server_start_time)
    return jsonify({
        "status": "alive",
        "ip": request.host.split(':')[0],
        "port": int(request.host.split(':')[1]) if ':' in request.host else 5000,
        "connected_miners": len(connected_miners),
        "uptime": uptime
    }), 200

def get_token_status(token: str):
    with open("token_store.json", "r") as f:
        token_data = json.load(f)
        if token_data["jwt"] == token and token_data["revoked"] == False:
            return "valid"
        else:
            return "invalid"

def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Modify the update_server_info function
def update_server_info():
    global connected_miners  # Ensure we're modifying the global variable
    while True:
        try:
            uptime = int(time.time() - server_start_time)
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            
            # Update connected_miners using get_miners function
            connected_miners = get_miners()
            
            server_info = {
                "status": "alive",
                "ip": ip,
                "port": args.port,  # Use the port from command line arguments
                "connected_miners": len(connected_miners),
                "uptime": uptime
            }
            for key, value in server_info.items():
                redis_key = f"server_info:{key}"
                redis_client.set(redis_key, str(value))
                app.logger.info(f"Updated Redis key: {redis_key} with value: {value}")
            time.sleep(60)  # Update every 60 seconds
        except Exception as e:
            app.logger.error(f"Error updating server info: {e}")
            time.sleep(60)  # Wait before trying again

# Add a function to check Redis connection
def check_redis_connection():
    try:
        redis_client.ping()
        app.logger.info("Redis connection successful")
        return True
    except redis.ConnectionError:
        app.logger.error("Unable to connect to Redis. Please check your Redis server and connection details.")
        return False

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

@app.before_request
def before_request():
    g.start_time = time.time()
    bt.logging.info(f"Received request: {request.method} {request.path}")
    bt.logging.info(f"Headers: {request.headers}")

@app.after_request
def after_request(response):
    diff = time.time() - g.start_time
    bt.logging.info(f"Request processed in {diff:.2f} seconds")
    bt.logging.info(f"Response status: {response.status}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    bt.logging.error("Unhandled exception occurred:")
    bt.logging.error(traceback.format_exc())
    return jsonify(error=str(e), traceback=traceback.format_exc()), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--public', action='store_true', help='Run server in public mode')
    args = parser.parse_args()

    if check_redis_connection():
        server_info_thread = threading.Thread(target=update_server_info, daemon=True)
        server_info_thread.start()  # Start the server info update thread
        app.logger.info(f"Starting Flask server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port)
    else:
        app.logger.error("Exiting due to Redis connection failure")
        sys.exit(1)

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