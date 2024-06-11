'''
functions to create signatures and verify signed synapses. we should use
data that is created on synapse creation.


'''
import bittensor as bt


def create_signature(data: str, wallet: bt.wallet) -> str:
    '''
    
    
    '''
    try:
        signature = wallet.hotkey.sign(data.encode().hex())
        return signature
    except Exception as e:
        print(f"Error signing data: {e}")
        return None


def verify_signature(hotkey: str, data: str, signature: str) -> bool:
    
    try:
        outcome = bt.Keypair(ss58_address=hotkey).verify(data, bytes.fromhex(signature))
        return outcome
    except Exception as e:
        print(f"Error verifying signature: {e}")
        return False