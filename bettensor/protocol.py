# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import datetime
import json
import typing
import uuid
import bittensor as bt
import bettensor
from bettensor.utils.sign_and_validate import create_signature
from uuid import UUID
import time
from pydantic import BaseModel, Field
import sqlite3




class Metadata(BaseModel):
    '''Synapse Metadata class, add more fields if needed'''
    synapse_id: UUID = Field(
        ...,
        description="UUID of the synapse"
    )
    neuron_uid: str = Field(
        ...,
        description="UUID of the serving neuron"
    )
    timestamp: str = Field(
        ...,
        description="Timestamp of the synapse"
    )
    signature: str = Field(
        ...,
        description="Signature of the serving neuron"
    )
    subnet_version: str = Field(
        ...,
        description="Subnet version of the neuron sending the synapse"
    )
    @classmethod
    def create(cls, wallet, subnet_version, neuron_uid):
        '''
        Creates a new metadata object
        Args:
            neuron_id: UUID
            signature: str
            subnet_id: str
        Returns:
            Metadata: A new metadata object to attach to a synapse
        '''
        synapse_id = uuid.uuid4()
        timestamp = datetime.datetime.now().isoformat()
        data_to_sign = f"{synapse_id}{timestamp}{neuron_uid}"
        signature = create_signature(wallet, data_to_sign)
        return cls(synapse_id=synapse_id, neuron_uid=neuron_uid, timestamp=timestamp, signature=signature, subnet_version=subnet_version)

    
class TeamGamePrediction(BaseModel):
    '''
    Data class from json. Will need to be modified in the future for more complex prediction types.
    '''
    predictionID: UUID = Field(
        ...,
        description="UUID of the prediction"
    )
    teamGameID: UUID = Field(
        ...,
        description="UUID of the team game"
    )
    minerID: UUID = Field(
        ...,
        description="UUID of the miner (coldkey/hotkey) that made the prediction"
    )
    predictionDate: str = Field(
        ...,
        description="Prediction date of the prediction"
    )
    predictedOutcome: str = Field(
        ...,
        description="Predicted outcome"
    )
    wager: float = Field(
        ...,
        description="Wager of the prediction"
    )
    teamAodds: float = Field(
        ...,
        description="Team A odds"
    )
    teamBodds: float = Field(
        ...,
        description="Team B odds"
    )
    tieOdds: float = Field(
        ...,
        description="Tie odds"
    )
    outcome: str = Field(
        ...,
        description="Outcome of prediction"
    )
    can_overwrite: bool = Field(
        ...,
        description="Can overwrite"
    )
    
    

class TeamGame(BaseModel):
    '''
    Data class from json. May need to be modified in the future for more complex prediction types
    '''
    id: UUID = Field(
        ...,
        description="UUID of the team game"
    )
    teamA: str = Field(
        ...,
        description="Team A"
    )
    teamB: str = Field(
        ...,
        description="Team B"
    )
    
    sport: str = Field(
        ...,
        description="Sport"
    )
    league: str = Field(
        ...,
        description="League"
    )
    externalId: str = Field(
        ...,
        description="External id of the team game"
    )
    createDate: str = Field(
        ...,
        description="Create date"
    )
    lastUpdateDate: str = Field(
        ...,
        description="Last update date"
    )
    eventStartDate: str = Field(
        ...,
        description="Event start date"
    )
    active: bool = Field(
        ...,
        description="Active"
    )
    outcome: str = Field(
        ...,
        description="Outcome"
    )
    teamAodds: float = Field(
        ...,
        description="Team A odds"
    )
    teamBodds: float = Field(
        ...,
        description="Team B odds"
    )
    tieOdds: float = Field(
        ...,
        description="Tie odds"
    )
    canTie: bool = Field(
        ...,
        description="Can tie"
    )
    
    



class Prediction(bt.Synapse):
    '''
    This class defines the synapse object for a miner prediction, consisting of a dictionary of TeamGamePrediction objects with a UUID as key.
    '''
    metadata: Metadata
    prediction_dict: typing.Dict[UUID, TeamGamePrediction]

    @classmethod
    def create(cls, metadata: Metadata, prediction_dict: typing.Dict[UUID, TeamGamePrediction]):
        '''
        Creates a new prediction synapse
        Args:
            metadata: Metadata
            prediction_dict: typing.Dict[UUID, TeamGamePrediction]
        Returns:
            Prediction: A new prediction synapse
        '''

        return cls(metadata=metadata, prediction_dict=prediction_dict)
    def deserialize(self):
        return self.prediction_dict, self.metadata

class GameData(bt.Synapse):
    '''
    This class defines the synapse object for game data, consisting of a dictionary of TeamGame objects with a UUID as key.
    '''
    metadata: Metadata
    gamedata_dict: typing.Dict[UUID, TeamGame]

    @classmethod
    def create(cls, metadata: Metadata, db_path):
        gamedata_dict = cls.fetch_game_data(metadata.timestamp, db_path)
        #metadata = cls.create_metadata()
        return cls(metadata=metadata, gamedata_dict=gamedata_dict)

    @staticmethod
    def fetch_game_data(current_timestamp, db_path) -> typing.Dict[UUID, TeamGame]:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        query = """
            SELECT id, teamA, teamB, sport, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds, canTie
            FROM game_data
            WHERE eventStartDate > ?
        """

        cursor.execute(query, (current_timestamp,))
        rows = cursor.fetchall()

        gamedata_dict = {}
        for row in rows:
            team_game = TeamGame(
                id=UUID(row[0]),
                teamA=row[1],
                teamB=row[2],
                sport=row[3],
                league=row[4],
                externalId=row[5],
                createDate=row[6],
                lastUpdateDate=row[7],
                eventStartDate=row[8],
                active=bool(row[9]),
                outcome=row[10],
                teamAodds=row[11],
                teamBodds=row[12],
                tieOdds=row[13],
                canTie=bool(row[14])
            )
            gamedata_dict[UUID(row[0])] = team_game

        connection.close()
        return gamedata_dict

    def deserialize(self):
        return self.gamedata_dict, self.metadata