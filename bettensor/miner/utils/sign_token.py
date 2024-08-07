import argparse
import requests
import bittensor as bt
from getpass import getpass
import json
import os

def get_jwt():
    print("Please enter the JWT token you received from the website:")
    jwt = input().strip()
    return jwt

def get_wallet():
    while True:
        print("\nNow, let's set up your wallet.")
        wallet_name = input("Enter your wallet name: ").strip()
        
        try:
            wallet = bt.wallet(name=wallet_name)
            
            if not wallet.coldkey_file.exists_on_device():
                print(f"Error: Coldkey file for wallet '{wallet_name}' does not exist.")
                print("Please make sure you've created a wallet with this name and it has a coldkey.")
                continue

            print("NOTICE: Signing this token will require your coldkey password.")
            print("This function uses the bittensor library and the password is not stored or recovered by this script.")
            
            return wallet
        except Exception as e:
            print(f"Error loading wallet: {e}")
            print("Please try again.")

def sign_jwt(wallet, jwt):
    message = jwt
    signature = wallet.coldkey.sign(message)
    return signature.hex()

def store_token(jwt, signature):
    token_data = {
        "jwt": jwt,
        "signature": signature,
        "revoked": False
    }
    with open("token_store.json", "w") as f:
        json.dump(token_data, f)

def main():
    print("Welcome to the Bettensor Token Signing Utility")
    
    jwt = get_jwt()
    wallet = get_wallet()

    try:
        signature = sign_jwt(wallet, jwt)
    except Exception as e:
        print(f"Error signing JWT: {e}")
        print("Please ensure your wallet is set up correctly and try again.")
        return

    store_token(jwt, signature)

    print("\nToken has been signed and stored successfully.")
    print("You can now use this signed token for authentication with the miner server.")

if __name__ == "__main__":
    main()