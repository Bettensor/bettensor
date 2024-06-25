"""
functions to create signatures and verify signed synapses. we should use
data that is created on synapse creation.


"""
import bittensor as bt


def create_signature(data: str, wallet: bt.wallet) -> str:
    """Signs the given data with the wallet hotkey

    Arguments:
        wallet:
            The wallet used to sign the Data
        data:
            Data to be signed

    Returns:
        signature:
            Signature of the key signing for the data
    """
    try:
        signature = wallet.hotkey.sign(data.encode()).hex()
        return signature
    except TypeError as e:
        bt.logging.error(
            f"Unable to sign data: {data} with wallet hotkey: {wallet.hotkey} due to error: {e}"
        )
        raise TypeError from e
    except AttributeError as e:
        bt.logging.error(
            f"Unable to sign data: {data} with wallet hotkey: {wallet.hotkey} due to error: {e}"
        )
        raise AttributeError from e


def verify_signature(hotkey: str, data: str, signature: str) -> bool:
    try:
        outcome = bt.Keypair(ss58_address=hotkey).verify(data, bytes.fromhex(signature))
        return outcome
    except Exception as e:
        print(f"Error verifying signature: {e}")
        return False
