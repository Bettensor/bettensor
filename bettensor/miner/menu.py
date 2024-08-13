import redis
import json
import os
import subprocess
import requests
import bittensor as bt
from bettensor.miner.interfaces.redis_interface import RedisInterface

# Set up logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

def get_redis_client():
    return RedisInterface(host=REDIS_HOST, port=REDIS_PORT)

def get_stored_token():
    if os.path.exists("token_store.json"):
        with open("token_store.json", "r") as f:
            return json.load(f)
    return None

def get_server_info(r):
    #bt.logging.debug("Attempting to retrieve server info from Redis")
    server_info = {}
    keys = ["status", "ip", "port", "connected_miners", "uptime"]
    for key in keys:
        value = r.get(f"server_info:{key}")
        if value:
            server_info[key] = value.decode('utf-8')
        else:
            bt.logging.warning(f"No value found for key: server_info:{key}")
    
    #bt.logging.debug(f"Retrieved server info: {server_info}")
    return server_info if server_info else None

def display_server_info(server_info):
    if not server_info:
        bt.logging.warning("Unable to retrieve server information.")
        bt.logging.warning("Please ensure that:")
        bt.logging.warning("1. Redis server is running")
        bt.logging.warning("2. The miner server is running and updating server information in Redis")
        bt.logging.warning("3. The Redis connection details (host, port, db) are correct")
        return

    print("\nServer Information:")
    print(f"Status: {server_info.get('status', 'Unknown')}")
    print(f"IP: {server_info.get('ip', 'Unknown')}")
    print(f"Port: {server_info.get('port', 'Unknown')}")
    print(f"Connected Miners: {server_info.get('connected_miners', 'Unknown')}")
    print(f"Uptime: {server_info.get('uptime', 'Unknown')}")
    print()

def prompt_loop():
    r = get_redis_client()
    if not r.connect():
        bt.logging.error("Failed to connect to Redis. Exiting.")
        return

    while True:
        server_info = get_server_info(r)
        display_server_info(server_info)

        print("1. Access CLI")
        print("2. Edit Model Parameters")
        print("3. Sign new token")
        print("4. Revoke current token")
        print("5. Check token status")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            subprocess.run(["python", "bettensor/miner/cli.py"])
        elif choice == "2":
            subprocess.run(["python", "bettensor/miner/model_params_tui.py"])
        elif choice == "3":
            subprocess.run(["python", "bettensor/miner/utils/sign_token.py"])
        elif choice == "4":
            token_data = get_stored_token()
            if token_data:
                r.publish("token_management", json.dumps({"action": "revoke", "data": token_data}))
                response = r.blpop("token_management_response", timeout=5)
                if response:
                    print(json.loads(response[1])["message"])
                else:
                    print("No response from server.")
            else:
                print("No token found.")
        elif choice == "5":
            token_data = get_stored_token()
            if token_data:
                r.publish("token_management", json.dumps({"action": "check", "data": token_data}))
                response = r.blpop("token_management_response", timeout=5)
                if response:
                    print(json.loads(response[1])["message"])
                else:
                    print("No response from server.")
            else:
                print("No token found.")
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    prompt_loop()