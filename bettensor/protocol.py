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

import json
import typing
import bittensor as bt
from uuid import UUID
import time

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2



class Metadata():
    '''Synapse Metadata class, add more fields if needed'''
    synapse_id : UUID
    neuron_id : UUID
    timestamp : time.time

    



class Prediction(bt.Synapse):
    '''
    This class defines the synapse object for a miner prediction, consisting of a dictionary of TeamGamePrediction objects with a UUID as key.
    '''
    metadata : Metadata
    prediction_dict : typing.Dict["key": UUID, "value": json.JSONDecoder]

    def deserialize(self) -> typing.Dict[UUID, json.JSONDecoder]:
       
        return self.prediction_dict


class GameData(bt.Synapse):
    '''
    This class defines the synapse object for a game data, consisting of a dictionary of TeamGameobjects with a UUID as key.
    '''
    metadata : Metadata
    gamedata_dict : typing.Dict["key": UUID, "value": json.JSONDecoder]

    def deserialize(self) -> typing.Dict[UUID, json.JSONDecoder]:
        return self.gamedata_dict