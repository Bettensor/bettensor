import unittest
import threading
import time
import requests
import json
import os
import logging
from bettensor.miner.interfaces.miner_interface_server import app, CHILD_CERT_PATH, fetch_and_store_child_cert, API_BASE_URL, CERT_ENDPOINT
import ssl
import jwt
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

JWT_SECRET = os.environ.get("JWT_SECRET", "test_secret_key")

class TestMinerInterfaceServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.token_store_path = os.path.join(os.path.dirname(__file__), 'test_token_store.json')
        with open(cls.token_store_path, 'w') as f:
            json.dump({}, f)
        
        os.environ['TOKEN_STORE_PATH'] = cls.token_store_path

        if os.path.exists(CHILD_CERT_PATH):
            os.remove(CHILD_CERT_PATH)

        def run_server():
            cert_path = fetch_and_store_child_cert()
            if cert_path:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                password = os.environ.get("CERT_PASSWORD", "default_password").encode()
                context.load_cert_chain(certfile=cert_path, password=password)
                app.config['TOKEN_STORE_PATH'] = cls.token_store_path
                app.run(host='localhost', port=5000, debug=False, use_reloader=False, ssl_context=context)
            else:
                logging.error('Failed to fetch certificate. Server not starting.')
                return
        
        cls.server_thread = threading.Thread(target=run_server)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        time.sleep(2)

    def test_certificate_fetching(self):
        cert_path = fetch_and_store_child_cert()
        self.assertIsNotNone(cert_path, "Certificate fetching failed")
        self.assertTrue(os.path.exists(CHILD_CERT_PATH), "Child certificate was not fetched")

        with open(CHILD_CERT_PATH, 'rb') as cert_file:
            cert_content = cert_file.read(100)
            logging.info(f"Certificate content (first 100 bytes): {cert_content}")

    def test_https_connection(self):
        try:
            test_token = self.generate_test_token()
            headers = {
                'Authorization': f'Bearer {test_token}',
                'Content-Type': 'application/json'
            }
            data = {
                "minerID": "test_miner",
                "predictions": [
                    {
                        "gameId": "123",
                        "homeTeamScore": 2,
                        "awayTeamScore": 1
                    }
                ]
            }
            response = requests.post('https://localhost:5000/submit_predictions', json=data, headers=headers, verify=False)
            logging.info(f"HTTPS response: {response.status_code}, {response.text}")
            self.assertEqual(response.status_code, 404, f"Unexpected status code: {response.status_code}")
            self.assertIn("Miner not found", response.text, "Unexpected error message")
        except requests.RequestException as e:
            self.fail(f"HTTPS request failed: {str(e)}")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.token_store_path):
            os.remove(cls.token_store_path)
        
        cls.server_thread.join(timeout=5)

        if os.path.exists(CHILD_CERT_PATH):
            os.remove(CHILD_CERT_PATH)

    def generate_test_token(self):
        payload = {
            'exp': datetime.now() + timedelta(days=1),
            'iat': datetime.now(),
            'sub': 'test_user'
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        
        with open(self.token_store_path, 'r+') as f:
            store = json.load(f)
            store['test_user'] = token
            f.seek(0)
            json.dump(store, f)
            f.truncate()
        
        return token

if __name__ == '__main__':
    unittest.main()