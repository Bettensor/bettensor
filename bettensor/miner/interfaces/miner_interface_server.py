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
        bt.logging.info(f"Entering token_required decorator for function: {f.__name__}")
        
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                token = auth_header

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        # Load the stored token
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        token_store_path = os.path.join(root_dir, 'token_store.json')
        bt.logging.info(f"Looking for token_store.json at: {token_store_path}")

        if not os.path.exists(token_store_path):
            bt.logging.error(f"token_store.json not found at {token_store_path}")
            return jsonify({'message': 'Token store not found!', 'path': token_store_path}), 500

        try:
            with open(token_store_path, 'r') as file:
                stored_token_data = json.load(file)
        except json.JSONDecodeError:
            bt.logging.error(f"Invalid JSON in token_store.json at {token_store_path}")
            return jsonify({'message': 'Invalid token store format!'}), 500

        stored_token = stored_token_data.get('jwt')
        if not stored_token:
            bt.logging.error("No 'jwt' field in token_store.json")
            return jsonify({'message': 'No valid token found in store!'}), 401

        if token != stored_token:
            bt.logging.warning("Provided token does not match stored token")
            return jsonify({'message': 'Invalid token!'}), 401

        if stored_token_data.get('revoked', False):
            bt.logging.warning("Token has been revoked")
            return jsonify({'message': 'Token has been revoked!'}), 401

        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except ExpiredSignatureError:
            bt.logging.warning("Token has expired")
            return jsonify({'message': 'Token has expired!'}), 401
        except InvalidTokenError:
            bt.logging.warning("Invalid token")
            return jsonify({'message': 'Invalid token!'}), 401

        bt.logging.info(f"Token verification successful. Calling {f.__name__}")
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
                    JOIN miner_active a ON m.miner_uid = CAST (a.miner_uid AS INTEGER)
                    WHERE a.last_active_timestamp > %s
                """, (datetime.utcnow() - timedelta(minutes=20),))
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

    # Generate self-signed certificate if it doesn't exist
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        bt.logging.info("Generating self-signed certificate...")
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        # Generate our key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Generate a self-signed certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u"San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Bettensor"),
            x509.NameAttribute(NameOID.COMMON_NAME, u"bettensor.com"),
        ])
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
            critical=False,
        ).sign(key, hashes.SHA256())

        # Write our certificate out to disk
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # Write our key out to disk
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    
    server_start_time = time.time()
    
    bt.logging.info(f"Starting Flask server on https://{args.host}:{args.port}")
    run_simple(args.host, args.port, app, ssl_context=context, use_reloader=False, threaded=True)