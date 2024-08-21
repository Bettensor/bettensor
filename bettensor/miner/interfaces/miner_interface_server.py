import os
import json
import ssl
import sys
import threading
import time
from flask import Flask, request, jsonify, g, make_response
from flask_cors import CORS
import redis
from werkzeug.serving import run_simple
import jwt 
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError, DecodeError
from bettensor.protocol import TeamGamePrediction
from functools import wraps
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
from datetime import datetime, timedelta

# Set up logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

CORS(app)

# Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'bettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensor')  # Change this in production
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

# Initialize Redis client
redis_interface = RedisInterface(host=REDIS_HOST, port=REDIS_PORT)
if not redis_interface.connect():
    bt.logging.error("Failed to connect to Redis. Exiting.")
    sys.exit(1)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per minute"]
)

# JWT token verification
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                token = auth_header

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401
            

        return f(*args, **kwargs)

    return decorated

@app.route('/submit_predictions', methods=['POST'])
@token_required
def submit_predictions():
    try:
        data = request.json
        miner_id = data.get('minerID')
        predictions = data.get('predictions', [])

        if not miner_id or not predictions:
            return jsonify({'error': 'Invalid request data'}), 400

        miner_uid = get_miner_uid(miner_id)
        if miner_uid is None:
            return jsonify({'error': 'Miner not found'}), 404

        message_id = str(uuid.uuid4())
        message = {
            'action': 'make_prediction',
            'message_id': message_id,
            'miner_id': miner_id,
            'predictions': predictions
        }

        channel = f'miner:{miner_uid}:{miner_id}'
        redis_client.publish(channel, json.dumps(message))

        response = None
        max_retries = 5
        for _ in range(max_retries):
            response = redis_client.get(f'response:{message_id}')
            if response:
                break
            time.sleep(1)

        if response:
            full_response = json.loads(response)
            server_response = {
                "amountLeft": full_response.get("amountLeft"),
                "tokenStatus": full_response.get("tokenStatus")
            }
            return jsonify(server_response)
        else:
            return jsonify({'error': 'No response from miner'}), 500

    except Exception as e:
        bt.logging.error(f"Error in submit_predictions: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/heartbeat', methods=['GET'])
@limiter.exempt
@token_required
def heartbeat():
    try:
        uptime = int(time.time() - server_start_time)
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(" ")[1] if auth_header and auth_header.startswith('Bearer ') else None
        
        token_status = get_token_status(token)
        active_miners = get_active_miners()
        miner_games = get_miner_games()
        
        response_data = {
            "tokenStatus": token_status,
            "miners": active_miners,
            "minerGames": miner_games
        }
        
        return jsonify(response_data)
    except Exception as e:
        bt.logging.error(f"Error in heartbeat: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

def get_miner_uid(hotkey: str):
    try:
        with psycopg2.connect(
                dbname="bettensor",
                user="root",
                password="bettensor_password",
                host="localhost"
            ) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT miner_uid FROM miner_stats WHERE miner_hotkey = %s", (hotkey,))
                miner = cur.fetchone()
            return miner['miner_uid'] if miner else None
    except Exception as e:
        bt.logging.error(f"Error in get_miner_uid: {str(e)}")
        return None

def get_active_miners():
    try:
        with psycopg2.connect(
            dbname="bettensor",
            user="root",
            password="bettensor_password",
            host="localhost"
        ) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT m.miner_uid, m.miner_hotkey, m.miner_rank, m.miner_cash
                    FROM miner_stats m
                    JOIN miner_active a ON m.miner_uid = a.miner_uid
                    WHERE a.last_active_timestamp > %s
                """, (datetime.now() - timedelta(minutes=20),))
                active_miners = cur.fetchall()
        return active_miners
    except Exception as e:
        bt.logging.error(f"Error fetching active miners: {str(e)}")
        return []

def get_miner_games():
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
        return miner_games
    except Exception as e:
        bt.logging.error(f"Error fetching miner games: {str(e)}")
        return []

def get_token_status(token: str):
    if not token:
        return "INVALID"
    
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return "VALID"
    except jwt.ExpiredSignatureError:
        return "EXPIRED"
    except jwt.InvalidTokenError:
        return "INVALID"

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Rate limit exceeded", message=str(e)), 429

@app.errorhandler(Exception)
def handle_exception(e):
    bt.logging.error(f"Unhandled exception: {str(e)}")
    bt.logging.error(traceback.format_exc())
    return jsonify(error=str(e), traceback=traceback.format_exc()), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()

    cert_path = os.path.join(os.path.dirname(__file__), 'cert.pem')
    key_path = os.path.join(os.path.dirname(__file__), 'key.pem')

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    
    server_start_time = time.time()
    
    bt.logging.info(f"Starting Flask server on https://{args.host}:{args.port}")
    run_simple(args.host, args.port, app, ssl_context=context, use_reloader=False, threaded=True)