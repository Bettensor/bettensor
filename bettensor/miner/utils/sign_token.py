import argparse
import requests
import bittensor as bt
from getpass import getpass

def get_wallet():
    while True:
        wallet_name = input("Enter your wallet name: ")
        wallet_hotkey = input("Enter your wallet hotkey name: ")
        try:
            wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
            return wallet
        except Exception as e:
            print(f"Error loading wallet: {e}")
            print("Please try again.")

def sign_jwt(wallet, jwt):
    message = bt.Keypair.create_message(jwt)
    signature = wallet.coldkey.sign(message)
    return signature.hex()

def main():
    parser = argparse.ArgumentParser(description="Central Server Authentication")
    parser.add_argument("--port", type=int, default=5000, help="Port of the Flask server")
    args = parser.parse_args()

    print("Welcome to the Bettensor Central Server Authentication")
    print("Please enter the JWT token you received from the website:")
    jwt = input().strip()

    wallet = get_wallet()

    signature = sign_jwt(wallet, jwt)

    # Get the server's IP address
    try:
        ip = requests.get('https://api.ipify.org').text
    except:
        ip = "Unable to determine IP"

    print("\nAuthentication Information:")
    print(f"Token: {jwt}")
    print(f"Signature: {signature}")
    print(f"IP: {ip}")
    print(f"Port: {args.port}")
    print("\nPlease use this information to connect to the central server.")

if __name__ == "__main__":
    main()
