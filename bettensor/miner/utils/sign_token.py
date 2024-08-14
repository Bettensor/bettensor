from typing import Any
import bittensor as bt
import argparse
from scalecodec.types import ScaleBytes
import json
import os
import requests

class ArgParserManager(argparse.ArgumentParser):
    def __init__(self, description=None):
        super().__init__(description=description)
        bt.subtensor.add_args(self)
        bt.logging.add_args(self)
        bt.wallet.add_args(self)
        self.add_argument("--message", type=str, help="Message to sign")
        self.add_argument("--wallet_name", type=str, help="Name of the wallet to use")
        self.config = bt.config(self)

def get_jwt():
    print("Please enter the JWT token you received from the website:")
    jwt = input().strip()
    return jwt

def get_wallet(config):
    while True:
        if config.wallet.name is None:
            config.wallet.name = input("Enter the name of the wallet to use: ").strip()

        wallet = bt.wallet(config=config)
        
        if not wallet.coldkey_file.exists_on_device():
            print(f"Error: Coldkey file for wallet '{config.wallet.name}' does not exist.")
            print("Please make sure you've created a wallet and it has a coldkey.")
            retry = input("Would you like to try another wallet name? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            config.wallet.name = None  # Reset wallet name to prompt again
        else:
            print("NOTICE: Signing this token will require your coldkey password.")
            print("This function uses the bittensor library and the password is not stored or recovered by this script.")
            return wallet

def sign_message(wallet: bt.wallet, message: Any):
    if isinstance(message, str):
        if message.startswith('0x'):
            message_bytes = bytes.fromhex(message[2:])
        else:
            message_bytes = message.encode()
    elif isinstance(message, ScaleBytes):
        message_bytes = bytes(message.data)
    elif isinstance(message, bytes):
        message_bytes = message
    else:
        raise ValueError("Message must be ScaleBytes, bytes, hex string, or regular string")

    signature = wallet.coldkey.sign(message_bytes)
    return signature.hex(), wallet.coldkey.ss58_address

def store_token(jwt, signature):
    token_data = {
        "jwt": jwt,
        "signature": signature,
        "revoked": False
    }
    with open("token_store.json", "w") as f:
        json.dump(token_data, f)

def get_ip():
    try:
        ip = requests.get('https://api.ipify.org').text
    except:
        try:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
        except:
            ip = "Unable to determine IP"
    return ip

def main():
    parser = ArgParserManager(description="Bettensor Token Signing Utility")
    config = bt.config(parser)
    args = parser.parse_args()

    if args.wallet_name:
        config.wallet.name = args.wallet_name

    if args.message:
        wallet = get_wallet(config)
        if wallet:
            signature, coldkey_address = sign_message(wallet, args.message)
            with open("signature.txt", "w") as f:
                f.write(f"Coldkey: {coldkey_address}\n")
                f.write(f"Message: 0x{args.message.encode().hex()}\n")
                f.write(f"Signature: 0x{signature}\n")
            print("Signature has been stored in signature.txt")
    else:
        print("Welcome to the Bettensor Token Signing Utility")
        
        jwt = get_jwt()
        wallet = get_wallet(config)

        if wallet:
            try:
                signature, coldkey_address = sign_message(wallet, jwt)
            except Exception as e:
                print(f"Error signing JWT: {e}")
                print("Please ensure your wallet is set up correctly and try again.")
                return

            store_token(jwt, signature)

            ip = get_ip()
            print(f"IP: {ip}")
            print(f"Port: 5000")
            print(f"Signature: {signature}")
            print(f"Coldkey Address: {coldkey_address}")
            print(f"Encoded JWT: 0x{jwt.encode().hex()}")

            print("\nToken has been signed and stored successfully.")
            print("You can now use this signed token for authentication with the miner server.")

if __name__ == "__main__":
    main()