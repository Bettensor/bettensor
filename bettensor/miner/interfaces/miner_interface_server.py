import os
import json
import ssl
import sys
import threading
import time
from flask import (
    Flask,
    request,
    jsonify,
    g,
    make_response,
    current_app,
)  # maybe remove current_app
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
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import pkcs12

bt.logging.set_trace(True)
bt.logging.set_debug(True)

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True

CORS(app)

# Configuration
JWT_SECRET = os.environ.get(
    "JWT_SECRET",
    "bettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensorbettensor",
)  # Change this in production
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

redis_interface = RedisInterface(host=REDIS_HOST, port=REDIS_PORT)
if not redis_interface.connect():
    bt.logging.error("Failed to connect to Redis. Exiting.")
    sys.exit(1)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

limiter = Limiter(
    key_func=get_remote_address, app=app, default_limits=["100 per minute"]
)

API_BASE_URL = "https://dev-bettensor-api.azurewebsites.net"
CERT_ENDPOINT = "/Certificate"
CHILD_CERT_PATH = os.path.join(os.path.dirname(__file__), "child_certificate.pem")


def fetch_and_store_child_cert():
    try:
        password = os.environ.get("CERT_PASSWORD", "default_password")
        ip_address = requests.get("https://api.ipify.org").text
        url = (
            f"{API_BASE_URL}{CERT_ENDPOINT}?ipAddress={ip_address}&password={password}"
        )

        bt.logging.info(f"Attempting to fetch certificate from {url}")
        response = requests.get(url, verify=True)

        bt.logging.info(f"Response status code: {response.status_code}")
        bt.logging.info(f"Response headers: {response.headers}")

        if response.status_code == 200:
            pfx_data = response.content
            password = password.encode()
            private_key, certificate, _ = pkcs12.load_key_and_certificates(
                pfx_data, password
            )

            cert_pem = certificate.public_bytes(encoding=serialization.Encoding.PEM)
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            with open(CHILD_CERT_PATH, "wb") as cert_file:
                cert_file.write(cert_pem + key_pem)

            bt.logging.info(
                f"Child certificate downloaded and saved to {CHILD_CERT_PATH}"
            )
            return CHILD_CERT_PATH
        else:
            bt.logging.error(
                f"Failed to download child certificate. Status code: {response.status_code}"
            )
            bt.logging.error(f"Response content: {response.text}")
            return None
    except Exception as e:
        bt.logging.error(
            f"Exception occurred while downloading child certificate: {str(e)}"
        )
        return None


def is_certificate_valid(cert_path):
    try:
        with open(cert_path, "rb") as cert_file:
            pfx_data = cert_file.read()
        password = os.environ.get("CERT_PASSWORD", "default_password").encode()
        private_key, certificate, _ = pkcs12.load_key_and_certificates(
            pfx_data, password
        )
        now = datetime.now()
        return certificate.not_valid_before <= now <= certificate.not_valid_after
    except Exception as e:
        bt.logging.error(f"Error checking certificate validity: {str(e)}")
        return False


if not os.path.exists(CHILD_CERT_PATH) or not is_certificate_valid(CHILD_CERT_PATH):
    fetch_and_store_child_cert()


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        bt.logging.info(f"Entering token_required decorator for function: {f.__name__}")

        token = None
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({"message": "Token is missing or invalid"}), 401

        if not token:
            return jsonify({"message": "Token is missing!"}), 401

        try:
            token_store_path = current_app.config.get(
                "TOKEN_STORE_PATH", os.environ.get("TOKEN_STORE_PATH")
            )
            bt.logging.info(f"Looking for token_store.json at: {token_store_path}")

            if not os.path.exists(token_store_path):
                bt.logging.error(f"token_store.json not found at {token_store_path}")
                return (
                    jsonify(
                        {"message": "Token store not found!", "path": token_store_path}
                    ),
                    500,
                )

            with open(token_store_path, "r") as token_file:
                token_store = json.load(token_file)

            if token not in token_store.values():
                return jsonify({"message": "Token is invalid!"}), 401

        except Exception as e:
            bt.logging.error(f"Error in token validation: {str(e)}")
            return jsonify({"message": "Token is invalid!"}), 401
        return f(*args, **kwargs)

    return decorated


@app.route("/submit_predictions", methods=["POST"])
@token_required
def submit_predictions():
    try:
        data = request.json
        miner_id = data.get("minerID")
        predictions = data.get("predictions", [])

        if not miner_id or not predictions:
            return jsonify({"error": "Invalid request data"}), 400

        miner_uid = get_miner_uid(miner_id)
        if miner_uid is None:
            return jsonify({"error": "Miner not found"}), 404

        message_id = str(uuid.uuid4())
        message = {
            "action": "make_prediction",
            "message_id": message_id,
            "miner_id": miner_id,
            "predictions": predictions,
        }

        channel = f"miner:{miner_uid}:{miner_id}"
        redis_client.publish(channel, json.dumps(message))

        response = None
        max_retries = 5
        for _ in range(max_retries):
            response = redis_client.get(f"response:{message_id}")
            if response:
                break
            time.sleep(1)

        if response:
            full_response = json.loads(response)
            server_response = {
                "amountLeft": full_response.get("amountLeft"),
                "tokenStatus": full_response.get("tokenStatus"),
            }
            return jsonify(server_response)
        else:
            return jsonify({"error": "No response from miner"}), 500

    except Exception as e:
        bt.logging.error(f"Error in submit_predictions: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/heartbeat", methods=["GET"])
@limiter.exempt
@token_required
def heartbeat():
    try:
        uptime = int(time.time() - server_start_time)
        auth_header = request.headers.get("Authorization")
        token = (
            auth_header.split(" ")[1]
            if auth_header and auth_header.startswith("Bearer ")
            else None
        )

        token_status = get_token_status(token)
        active_miners = get_active_miners()
        miner_games = get_miner_games()

        response_data = {
            "tokenStatus": token_status,
            "miners": active_miners,
            "minerGames": miner_games,
        }

        return jsonify(response_data)
    except Exception as e:
        bt.logging.error(f"Error in heartbeat: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


def get_miner_uid(hotkey: str):
    try:
        with psycopg2.connect(
            dbname="bettensor",
            user="root",
            password="bettensor_password",
            host="localhost",
        ) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT miner_uid FROM miner_stats WHERE miner_hotkey = %s",
                    (hotkey,),
                )
                miner = cur.fetchone()
            return miner["miner_uid"] if miner else None
    except Exception as e:
        bt.logging.error(f"Error in get_miner_uid: {str(e)}")
        return None


def get_active_miners():
    try:
        with psycopg2.connect(
            dbname="bettensor",
            user="root",
            password="bettensor_password",
            host="localhost",
        ) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT m.miner_uid, m.miner_hotkey, m.miner_rank, m.miner_cash
                    FROM miner_stats m
                    JOIN miner_active a ON m.miner_uid = CAST (a.miner_uid AS INTEGER)
                    WHERE a.last_active_timestamp > %s
                """,
                    (datetime.now() - timedelta(minutes=20),),
                )
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
            host="localhost",
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT externalID
                    FROM games
                    WHERE CAST(eventStartDate AS TIMESTAMP) > NOW()
                    ORDER BY CAST(eventStartDate AS TIMESTAMP) ASC
                """
                )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    args = parser.parse_args()

    cert_path = os.path.join(os.path.dirname(__file__), "child_certificate.pem")
    key_path = os.path.join(os.path.dirname(__file__), "private_key.pem")

    if not os.path.exists(cert_path) or not is_certificate_valid(cert_path):
        bt.logging.info("Downloading child certificate...")
        cert_path = fetch_and_store_child_cert()
        if not cert_path:
            bt.logging.error("Failed to download child certificate. Exiting.")
            sys.exit(1)

    if not os.path.exists(key_path):
        bt.logging.info("Generating a new private key...")
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        with open(key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
    else:
        with open(key_path, "rb") as f:
            key_data = f.read()
        key = serialization.load_pem_private_key(
            key_data,
            password=None,
        )

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    with open(cert_path, "rb") as cert_file:
        pfx_data = cert_file.read()
    password = os.environ.get("CERT_PASSWORD", "default_password").encode()
    context.load_cert_chain(certfile=cert_path, password=password)

    server_start_time = time.time()

    bt.logging.info(f"Starting Flask server on https://{args.host}:{args.port}")
    run_simple(
        args.host,
        args.port,
        app,
        ssl_context=context,
        use_reloader=False,
        threaded=True,
    )
