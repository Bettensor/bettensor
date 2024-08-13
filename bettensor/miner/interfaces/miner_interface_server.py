import os
import json
import sys
import threading
import time
from flask import Flask, request, jsonify, g, make_response
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
    default_limits=["20 per minute"]
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
        print("==== Token Required Decorator ====")
        print(f"Request Method: {request.method}")
        print(f"Request URL: {request.url}")
        print("Request Headers:")
        for header, value in request.headers:
            print(f"  {header}: {value}")
        print("Request Data:")
        print(request.get_data(as_text=True))
        print("================================")

        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            print(f"Authorization header found: {auth_header}")
            try:
                token = auth_header.split(" ")[1]
                print(f"Extracted token: {token[:10]}...")
            except IndexError:
                token = auth_header
                print(f"No 'Bearer ' prefix, using full header as token: {token[:10]}...")
        else:
            print("No Authorization header found")

        if not token:
            print("Token is missing")
            return jsonify({'message': 'Token is missing!'}), 401

        # Check if the token matches the one in token_store.json
        if os.path.exists("token_store.json"):
            try:
                with open("token_store.json", "r") as file:
                    stored_data = json.load(file)
                    stored_jwt = stored_data.get("jwt")
                    if stored_jwt and stored_jwt != token:
                        print("Received token does not match the stored token")
                        return jsonify({'message': 'Invalid token!'}), 401
                    else:
                        print("Token matches stored token")
            except json.JSONDecodeError as e:
                print(f"Error decoding token_store.json: {str(e)}")
                return jsonify({'message': 'Error reading stored token!'}), 500
        else:
            print("token_store.json not found")

        try:
            # Decode the token without verifying the signature
            decoded = pyjwt.decode(token, options={"verify_signature": False})
            print(f"Decoded token: {decoded}")
            
            # Now verify the signature separately
            pyjwt.decode(token, config.JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
            print("Token signature successfully verified")
        except ExpiredSignatureError:
            print("Token has expired")
            return jsonify({'message': 'Token has expired!'}), 401
        except (InvalidTokenError, pyjwt.DecodeError) as e:
            print(f"Invalid token: {str(e)}")
            print(f"JWT_SECRET used: {config.JWT_SECRET}")
            return jsonify({'message': 'Token is invalid!'}), 401

        print("Token validation successful")
        return f(*args, **kwargs)

    return decorated

def get_miner_uids():
    pass

@app.route('/submit_predictions', methods=['POST'])
@token_required
def submit_predictions():
    print("==== Submit Predictions Function ====")
    print(f"Request Method: {request.method}")
    print(f"Request URL: {request.url}")
    print("Request Headers:")
    for header, value in request.headers:
        print(f"  {header}: {value}")
    print("Request Data:")
    print(request.get_data(as_text=True))
    print("=====================================")

    try:
        print("Attempting to parse request JSON")
        data = request.json
        print(f"Received data: {json.dumps(data, indent=2)}")

        print("Extracting miner ID and predictions from data")
        miner_id = data.get('minerID')
        predictions = data.get('predictions', [])

        print(f"Miner ID: {miner_id}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Predictions: {json.dumps(predictions, indent=2)}")

        if not miner_id or not predictions:
            print("Invalid request data: missing miner ID or predictions")
            return jsonify({'error': 'Invalid request data'}), 400

        print("Attempting to get miner UID")
        miner_uid = get_miner_uid(miner_id)
        print(f"Miner UID: {miner_uid}")

        if miner_uid is None:
            print(f"Miner not found for ID: {miner_id}")
            return jsonify({'error': 'Miner not found'}), 404

        print("Generating unique message ID")
        message_id = str(uuid.uuid4())
        print(f"Generated message ID: {message_id}")

        print("Preparing message for Redis")
        message = {
            'action': 'make_prediction',
            'message_id': message_id,
            'miner_id': miner_id,
            'predictions': predictions
        }
        print(f"Prepared message: {json.dumps(message, indent=2)}")

        print("Preparing to publish message to Redis")
        channel = f'miner:{miner_uid}:{miner_id}'
        print(f"Publishing to channel: {channel}")
        
        print("Attempting to publish message to Redis")
        try:
            publish_result = redis_client.publish(channel, json.dumps(message))
            print(f"Publish result: {publish_result}")
        except Exception as redis_error:
            print(f"Error publishing to Redis: {str(redis_error)}")
            print(traceback.format_exc())
            return jsonify({'error': 'Error communicating with Redis'}), 500

        print(f"Waiting for response with key: response:{message_id}")
        response = None
        max_retries = 5
        for i in range(max_retries):
            print(f"Attempt {i+1} to get response from Redis")
            response = redis_client.get(f'response:{message_id}')
            if response:
                print(f"Received response: {response}")
                break
            else:
                print(f"No response received, waiting 1 second before retry")
                time.sleep(1)

        if response:
            print("Parsing and returning response")
            full_response = json.loads(response)
            # Extract only the necessary fields for the server response
            server_response = {
                "amountLeft": full_response.get("amountLeft"),
                "tokenStatus": full_response.get("tokenStatus")
            }
            return jsonify(server_response)
        else:
            print("No response received from miner after all retries")
            return jsonify({'error': 'No response from miner'}), 500

    except json.JSONDecodeError as json_error:
        print(f"JSON Decode Error: {str(json_error)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Invalid JSON in request'}), 400
    except redis.RedisError as redis_error:
        print(f"Redis Error: {str(redis_error)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Redis communication error'}), 500
    except Exception as e:
        print(f"Unexpected Exception in submit_predictions: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

def get_miner_uid(hotkey: str):
    print(f"Getting miner UID for hotkey: {hotkey}")
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
                print(f"Found miner UID: {miner['miner_uid']}")
                return miner['miner_uid']
            else:
                print(f"No miner found for hotkey: {hotkey}")
                return None
    except Exception as e:
        print(f"Error in get_miner_uid: {str(e)}")
        print(traceback.format_exc())
        raise

def get_miners() -> list:
    print("Attempting to retrieve miners from database")
    try:
        conn = psycopg2.connect(
            dbname="bettensor",
            user="root",
            password="bettensor_password",
            host="localhost"
        )
        print("Database connection established")
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
        print(f"Retrieved {len(miners)} miners from database")
        return [dict(miner) for miner in miners]
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in get_miners: {str(e)}")
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
@limiter.exempt
@token_required
def heartbeat():
    print("==== Heartbeat Function ====")
    print(f"Request Method: {request.method}")
    print(f"Request URL: {request.url}")
    print("Request Headers:")
    for header, value in request.headers:
        print(f"  {header}: {value}")
    print("Request Data:")
    print(request.get_data(as_text=True))
    print("=============================")

    start_time = time.time()
    
    uptime = int(time.time() - server_start_time)
    print(f"Server uptime: {uptime} seconds")
    
    token_status = get_token_status(request.headers.get('Authorization'))
    print(f"Token status: {token_status}")

    # Request upcoming game IDs
    message_id = str(uuid.uuid4())
    print(f"Requesting upcoming game IDs with message_id: {message_id}")
    redis_client.publish('game_requests', json.dumps({
        'action': 'get_upcoming_game_ids',
        'message_id': message_id
    }))
    
    # Wait for response
    response_key = f'response:{message_id}'
    upcoming_game_ids = []
    max_wait = 5  # Maximum wait time in seconds
    wait_start_time = time.time()
    
    while time.time() - wait_start_time < max_wait:
        response = redis_client.get(response_key)
        if response:
            print(f"Received response for upcoming game IDs: {response}")
            upcoming_game_ids = json.loads(response)
            redis_client.delete(response_key)
            break
        time.sleep(0.1)
    
    if not upcoming_game_ids:
        print("No upcoming game IDs received within the timeout period")
    
    miners = get_miners()
    print(f"Retrieved {len(miners)} miners")
    
    # Fetch miner games
    miner_games = get_miner_games()
    print(f"Retrieved {len(miner_games)} miner game IDs")
    
    response_data = {
        "tokenStatus": token_status,
        "miners": miners,
        "minerGames": miner_games
    }
    
    print(f"Heartbeat response prepared: {json.dumps(response_data, indent=2)}")
    
    end_time = time.time()
    print(f"Heartbeat request processed in {end_time - start_time:.2f} seconds")

    response = make_response(jsonify(response_data))
    
    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    
    print("Response Headers:")
    for header, value in response.headers:
        print(f"  {header}: {value}")
    
    return response

def get_miner_games():
    print("Fetching miner games")
    try:
        with psycopg2.connect(
            dbname="bettensor",
            user="root",
            password="bettensor_password",
            host="localhost"
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT externalID
                    FROM games
                    WHERE CAST(eventStartDate AS TIMESTAMP) > NOW()
                    ORDER BY CAST(eventStartDate AS TIMESTAMP) ASC
                """)
                miner_games = [row[0] for row in cur.fetchall()]
        print(f"Fetched {len(miner_games)} miner game IDs")
        return miner_games
    except Exception as e:
        print(f"Error fetching miner games: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

def get_token_status(token: str):
    with open("token_store.json", "r") as f:
        token_data = json.load(f)
        if token_data["jwt"] == token and token_data["revoked"] == False:
            return "VALID"
        else:
            return "INVALID"

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
    print(f"Received request: {request.method} {request.path}")
    print(f"Headers: {request.headers}")

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        diff = time.time() - g.start_time
        print(f"Request processed in {diff:.2f} seconds")
    print(f"Response status: {response.status}")
    return response

@app.errorhandler(429)
def ratelimit_handler(e):
    print(f"Rate limit exceeded: {e}")
    return jsonify(error="Rate limit exceeded", message=str(e)), 429

@app.errorhandler(401)
def unauthorized(error):
    print("==== 401 Unauthorized Error ====")
    print(f"Error: {error}")
    print(f"Request Method: {request.method}")
    print(f"Request URL: {request.url}")
    print("Request Headers:")
    for header, value in request.headers:
        print(f"  {header}: {value}")
    print("Request Data:")
    print(request.get_data(as_text=True))
    print("================================")
    
    response = jsonify({'error': 'Unauthorized', 'message': str(error)})
    response.status_code = 401
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    print("Unhandled exception occurred:")
    print(traceback.format_exc())
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
        print(f"Starting Flask server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port)
    else:
        print("Exiting due to Redis connection failure")
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
