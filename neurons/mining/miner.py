# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import datetime
import sqlite3
import json
import time
import typing
from uuid import UUID
import uuid
from bettensor.protocol import TeamGamePrediction
import bittensor as bt

# Bittensor Miner Template:
import bettensor

# import base miner class which takes care of most of the boilerplate
from bettensor.base.miner import BaseMinerNeuron
from bittensor import logging, synapse


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        try:
            db = sqlite3.connect('./miner.db')
        except:
            bt.logging.error("Failed to connect to local database")
            raise Exception("Failed to connect to local database")
        
        cursor = db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (gameId TEXT, dateTime TEXT, wager INTEGER, predictedOutcome INTEGER)''')
        games_dict = {}
        predictions_dict = {}
        cash = 1000

        try:
            db = sqlite3.connect('./miner.db')
        except:
            bt.logging.error("Failed to connect to local database")
            raise Exception("Failed to connect to local database")
        
        cursor = db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (gameId TEXT, dateTime TEXT, wager INTEGER, predictedOutcome INTEGER)''')
        games_dict = {}
        predictions_dict = {}
        cash = 1000


    async def forward(
        self, gamedata: bettensor.protocol.GameData, games_dict, predictions_dict
    ) -> bettensor.protocol.Prediction:
        """
        Takes an incoming synapse of game data, and runs CLI for user to submit predictions. Submits UUID's of predicted games to chain, waits until acceptance, then returns synapse with prediction data to validator. If games have already been predicted and committed to chain, 
        return synapse to any "new" validator( one that has not yet communicated with this miner about recent games). If all validators have already been notified, wait until a new game is available. (this logic will happen above the forward function, we don't want to call it 
        repeatedly if we've already submitted predictions for the current period).

        Args:
            GameData : The synapse object containing the game data.

        Returns:
            synapse : Synapse object with prediction data. Must be compared to on chain data before acceptance (Validator side)
        """


        deserialized_synapse = bettensor.protocol.GameData.deserialize(synapse.data)

        for game in deserialized_synapse.data:
            if game not in bettensor.protocol.GameData.games_dict:
                bettensor.protocol.GameData.games_dict[game[0]] = game
        

        response = bettensor.protocol.Prediction()
        

        return response

        
        

    async def blacklist(
        self, synapse: bettensor.protocol.Dummy
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bettensor.protocol.Dummy) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority



def reset_daily_cash():
    pass
        



def construct_prediction_json(prediction_dict: typing.Dict[UUID, TeamGamePrediction]) -> json.JSONDecoder:
    '''
    Method to take a dictionary of predictions and construct and validate a json object 
    that gets sent to validators
    '''
   
    prediction_json = json.dumps(prediction_dict)
    return prediction_json




# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
