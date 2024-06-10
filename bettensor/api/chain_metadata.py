import bittensor as bt
from bettensor.protocol import TeamGamePrediction
from typing import Optional




class CommitPredictionMetadata():

    def __init__(
            self,
            subtensor: bt.Subtensor,
            hotkey: str,
            wallet: Optional[bt.wallet] = None,
            subnet_uid : int = 0 # TODO: set in a constants file/config file
    ):
         self.subtensor = subtensor
         self.wallet = wallet
         self.subnet_uid = subnet_uid



    async def commit_predictions(prediction_dict: dict[TeamGamePrediction], hotkey: str) -> None:
        '''
        method to commit miner predictions to the bittensor chain
        '''
    # TODO: Strip dictionary to game_id, and prediction values.
    # TODO: Create a hashed string to commit to chain

     
    pass


    async def test_commit_predictions(self) -> None:
        pass


class ReadPredictionMetadata():

    def __init__(
            self,
            subtensor: bt.Subtensor,
            hotkey: str,
            wallet: Optional[bt.wallet] = None,
            subnet_uid : int = 0 # TODO: set in a constants file/config file
                 
                 
                 
                 ) :
        self.subtensor = subtensor
        self.hotkey = hotkey
        self.wallet = wallet
        self.subnet_uid = subnet_uid

        pass
