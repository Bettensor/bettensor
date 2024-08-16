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

from datetime import datetime, timedelta, timezone
import json
from typing import Optional, Dict
import uuid
import bittensor as bt
import bettensor
from bettensor.utils.sign_and_validate import create_signature
from uuid import UUID
import time
from pydantic import BaseModel, Field
import sqlite3


class MinerStats(BaseModel):
    """
    This class defines the miner stats object
    """

    miner_hotkey: str = Field(..., description="Hotkey of the miner")
    miner_coldkey: str = Field(..., description="Coldkey of the miner")
    miner_uid: str = Field(..., description="Current UID of the miner")
    miner_rank: int = Field(..., description="Current rank of the miner")
    miner_cash: float = Field(..., description="Current cash of the miner")
    miner_current_incentive: float = Field(..., description="Current incentive of the miner")
    miner_last_prediction_date: Optional[str] = Field(None, description="Date of the last prediction of the miner")
    miner_lifetime_earnings: float = Field(..., description="Lifetime earnings of the miner")
    miner_lifetime_wager: float = Field(..., description="Lifetime wager of the miner")
    miner_lifetime_predictions: int = Field(..., description="Lifetime predictions of the miner")
    miner_lifetime_wins: int = Field(..., description="Lifetime wins of the miner")
    miner_lifetime_losses: int = Field(..., description="Lifetime losses of the miner")
    miner_win_loss_ratio: float = Field(..., description="Win loss ratio of the miner")
    miner_status: str = Field(..., description="Status of the miner")

    @classmethod
    def create(cls, row):
        """
        takes a row from the miner_stats table and returns a MinerStats object
        """
        miner_hotkey = row[0]
        miner_coldkey = row[1]
        miner_uid = row[2]
        miner_rank = row[3]
        miner_cash = row[4]
        miner_current_incentive = row[5]
        miner_last_prediction_date = row[6]
        miner_lifetime_earnings = row[7]
        miner_lifetime_wager = row[8]
        miner_lifetime_predictions = row[9]
        miner_lifetime_wins = row[10]
        miner_lifetime_losses = row[11]
        miner_win_loss_ratio = row[12]
        miner_status = row[13]
        return cls(
            miner_hotkey=miner_hotkey,
            miner_coldkey=miner_coldkey,
            miner_uid=miner_uid,
            miner_rank=miner_rank,
            miner_cash=miner_cash,
            miner_current_incentive=miner_current_incentive,
            miner_last_prediction_date=miner_last_prediction_date,
            miner_lifetime_earnings=miner_lifetime_earnings,
            miner_lifetime_wager=miner_lifetime_wager,
            miner_lifetime_predictions=miner_lifetime_predictions,
            miner_lifetime_wins=miner_lifetime_wins,
            miner_lifetime_losses=miner_lifetime_losses,
            miner_win_loss_ratio=miner_win_loss_ratio,
            miner_status=miner_status,
        )


class Metadata(BaseModel):
    """Synapse Metadata class, add more fields if needed"""

    synapse_id: str = Field(..., description="UUID of the synapse")
    neuron_uid: str = Field(..., description="UUID of the serving neuron")
    timestamp: str = Field(..., description="Timestamp of the synapse")
    signature: str = Field(..., description="Signature of the serving neuron")
    subnet_version: str = Field(
        ..., description="Subnet version of the neuron sending the synapse"
    )
    synapse_type: str = Field(
        ..., description="Type of the synapse | 'prediction' or 'game_data'"
    )

    @classmethod
    def create(cls, wallet: bt.wallet, subnet_version, neuron_uid, synapse_type):
        """
        Creates a new metadata object
        Args:
            neuron_id: UUID
            signature: str
            subnet_id: str
        Returns:
            Metadata: A new metadata object to attach to a synapse
        """
        synapse_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        data_to_sign = f"{synapse_id}{timestamp}{neuron_uid}"
        signature = create_signature(data_to_sign, wallet)
        bt.logging.debug(
            f"Creating Metadata with synapse_id: {synapse_id}, neuron_uid: {neuron_uid}, timestamp: {timestamp}, signature: {signature}, subnet_version: {subnet_version}"
        )
        return Metadata(
            synapse_id=synapse_id,
            neuron_uid=neuron_uid,
            timestamp=timestamp,
            signature=signature,
            subnet_version=subnet_version,
            synapse_type=synapse_type,
        )


class TeamGamePrediction(BaseModel):
    """
    Data class from json. Will need to be modified in the future for more complex prediction types.
    """

    predictionID: str = Field(..., description="UUID of the prediction")
    teamGameID: str = Field(..., description="UUID of the team game")
    minerID: str = Field(..., description="UUID or UID of the miner that made the prediction")
    predictionDate: str = Field(..., description="Prediction date of the prediction")
    predictedOutcome: str = Field(..., description="Predicted outcome")
    teamA: Optional[str] = Field(None, description="Team A")
    teamB: Optional[str] = Field(None, description="Team B")
    wager: float = Field(..., description="Wager of the prediction")
    teamAodds: float = Field(..., description="Team A odds")
    teamBodds: float = Field(..., description="Team B odds")
    tieOdds: Optional[float] = Field(None, description="Tie odds")
    outcome: str = Field(..., description="Outcome of prediction")
    can_overwrite: Optional[bool] = Field(True, description="Can overwrite")


class TeamGame(BaseModel):
    """
    Data class from json. May need to be modified in the future for more complex prediction types
    """

    id: str = Field(..., description="ID of the team game")
    teamA: str = Field(..., description="Team A")
    teamB: str = Field(..., description="Team B")

    sport: str = Field(..., description="Sport")
    league: str = Field(..., description="League")
    externalId: str = Field(..., description="External id of the team game")
    createDate: str = Field(..., description="Create date")
    lastUpdateDate: str = Field(..., description="Last update date")
    eventStartDate: str = Field(..., description="Event start date")
    active: bool = Field(..., description="Active")
    outcome: str = Field(..., description="Outcome")
    teamAodds: float = Field(..., description="Team A odds")
    teamBodds: float = Field(..., description="Team B odds")
    tieOdds: float = Field(..., description="Tie odds")
    canTie: bool = Field(..., description="Can tie")


class GameData(bt.Synapse):
    """
    This class defines the synapse object for game data, consisting of a dictionary of TeamGame objects with a UUID as key.
    """

    metadata: Optional[Metadata]
    gamedata_dict: Optional[Dict[str, TeamGame]]
    prediction_dict: Optional[Dict[str, TeamGamePrediction]]
    error: Optional[str]

    @classmethod
    def create(
        cls,
        db_path: str,
        wallet: bt.wallet,
        subnet_version: str,
        neuron_uid: int,  # Note: This is an int
        synapse_type: str,
        prediction_dict: Dict[str, TeamGamePrediction] = None,
    ):
        metadata = Metadata.create(
            wallet=wallet,
            subnet_version=subnet_version,
            neuron_uid=str(neuron_uid),  # Convert to string here
            synapse_type=synapse_type,
        )
        if synapse_type == "prediction":
            gamedata_dict = None
        else:
            gamedata_dict = cls.fetch_game_data(metadata.timestamp, db_path)
        return cls(
            metadata=metadata,
            gamedata_dict=gamedata_dict,
            prediction_dict=prediction_dict,
            synapse_type=synapse_type,
        )

    @staticmethod
    def fetch_game_data(current_timestamp, db_path) -> Dict[str, TeamGame]:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Calculate timestamp for 5 days ago
        fifteen_days_ago = (datetime.fromisoformat(current_timestamp) - timedelta(days=15)).isoformat()

        query = """
            SELECT id, teamA, teamB, sport, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds, canTie
            FROM game_data
            WHERE eventStartDate > ? OR (eventStartDate BETWEEN ? AND ?)
        """

        cursor.execute(query, (current_timestamp, fifteen_days_ago, current_timestamp))
        rows = cursor.fetchall()

        gamedata_dict = {}
        for row in rows:
            team_game = TeamGame(
                id=row[0],
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
                canTie=bool(row[14]),
            )
            gamedata_dict[row[0]] = team_game

        connection.close()
        return gamedata_dict

    def deserialize(self):
        return self.gamedata_dict, self.prediction_dict, self.metadata