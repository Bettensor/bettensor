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
from pydantic import BaseModel

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



class Metadata(BaseModel):
    '''Synapse Metadata class, add more fields if needed'''
    synapse_id : UUID
    neuron_id : UUID
    timestamp : str # TODO: Match timestamp format

    
class TeamGamePrediction(BaseModel):
    '''
    Data class from json. May need to be modified in the future for more complex prediction types
    '''
    pred_id : UUID # id of the prediction
    teamGameId : UUID # id of the team game
    league: int
    minerId : UUID # id of the miner (coldkey/hotkey) that made the prediction
    predictionDate : str # TODO: Match timestamp format
    predictedOutcome : str
    wager: float
    
    

class TeamGame(BaseModel):
    '''
    Data class from json. May need to be modified in the future for more complex prediction types
    '''
    id : UUID # id of the team game
    teamA : str
    teamB : str
    teamAodds: float
    teamBodds: float
    sport : str
    league : str
    eventDescription : str
    externalId : str # external id of the team game
    createDate : str # TODO: Match timestamp format
    lastUpdateDate : str # TODO: Match timestamp format 
    eventStartDate: str
    active : bool
    outcome : str
    
    



class Prediction(bt.Synapse, BaseModel):
    '''
    This class defines the synapse object for a miner prediction, consisting of a dictionary of TeamGamePrediction objects with a UUID as key.
    '''
    #dummy_data: int
    metadata : Metadata
    prediction_dict: typing.Dict[UUID, TeamGamePrediction]
    def deserialize(self) -> typing.Dict[UUID, TeamGamePrediction]:
    #def deserialize(self) -> int:
        
        #return self.dummy_data
        return self.prediction_dict, self.metadata


class GameData(bt.Synapse, BaseModel):
    '''
    This class defines the synapse object for game data, consisting of a dictionary of TeamGame object with a UUID as key.
    '''
    metadata : Metadata
    gamedata_dict: typing.Dict[UUID, TeamGame]

    def __init__(self, current_timestamp, db_url):
        super().__init__()
        self.current_timestamp = current_timestamp
        self.db_path = db_path
        self.gamedata_dict = self.fetch_game_data()
    
    def fetch_game_data(self) -> typing.Dict[UUID, TeamGame]:
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
    
        query = """
            SELECT id, teamA, teamB, sport, league, eventDescription, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome 
            FROM game_data
            WHERE eventStartDate > ?
            """
            
        cursor.execute(query, (self.current_timestamp,))
        rows = cursor.fetchall()

        gamedata_dict = {}
        for row in rows:
            team_game = TeamGame(
                    id=UUID(row[0]),
                    teamA=row[1],
                    teamB=row[2],
                    sport=row[3],
                    league=row[4],
                    eventDescription=row[5],
                    externalId=row[6],
                    createDate=row[7],
                    lastUpdateDate=row[8],
                    eventStartDate=row[9],
                    active=bool(row[10]),
                    outcome=row[11]
                )
            gamedata_dict[UUID(row[0])] = team_game

        connection.close()
        return gamedata_dict



    def deserialize(self) -> typing.Dict[UUID, TeamGame]:
        return self.gamedata_dict, self.metadata
