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
from typing import Optional, Dict
import uuid
import bittensor as bt
from pydantic import BaseModel, Field
import sqlite3


class MinerStats(BaseModel):
    """
    This class defines the miner stats object.
    """

    # Identifier Stats
    miner_hotkey: str = Field(..., description="Hotkey of the miner")
    miner_coldkey: str = Field(..., description="Coldkey of the miner")
    miner_uid: str = Field(..., description="Current UID of the miner")
    miner_rank: int = Field(..., description="Current rank of the miner")
    miner_status: str = Field(..., description="Status of the miner (Active/Inactive)")

    # Current Scoring Stats
    miner_cash: float = Field(..., description="Current cash of the miner")
    miner_current_incentive: float = Field(
        ..., description="Current incentive of the miner"
    )
    miner_current_tier: int = Field(..., description="Current tier of the miner")
    miner_current_scoring_window: int = Field(
        ..., description="Current scoring window of the miner"
    )
    miner_current_composite_score: float = Field(
        ...,
        description="Current composite score of the miner, as calculated over the current scoring window",
    )
    miner_current_sharpe_ratio: float = Field(
        ...,
        description="Current sharpe ratio of the miner, as calculated over the current scoring window",
    )
    miner_current_sortino_ratio: float = Field(
        ...,
        description="Current sortino ratio of the miner, as calculated over the current scoring window",
    )
    miner_current_roi: float = Field(
        ...,
        description="Current roi of the miner, as calculated over the current scoring window",
    )
    miner_current_clv_avg: float = Field(
        ...,
        description="Current clv avg of the miner, as calculated over the current scoring window",
    )

    # Lifetime Stats
    miner_last_prediction_date: Optional[str] = Field(
        None, description="Date of the last prediction of the miner"
    )
    miner_lifetime_earnings: float = Field(
        ..., description="Lifetime earnings of the miner"
    )
    miner_lifetime_wager_amount: float = Field(
        ..., description="Lifetime wager amount of the miner"
    )
    miner_lifetime_roi: float = Field(
        ..., description="Lifetime roi of the miner"
    )
    miner_lifetime_predictions: int = Field(
        ..., description="Lifetime predictions of the miner"
    )
    miner_lifetime_wins: int = Field(..., description="Lifetime wins of the miner")
    miner_lifetime_losses: int = Field(..., description="Lifetime losses of the miner")
    miner_win_loss_ratio: float = Field(..., description="Win loss ratio of the miner")

    @classmethod
    def create(cls, row):
        """
        takes a row from the miner_stats table and returns a MinerStats object
        """
        miner_hotkey = row[0]
        miner_coldkey = row[1]
        miner_uid = row[2]
        miner_rank = row[3]
        miner_status = row[4]
        miner_cash = row[5]
        miner_current_incentive = row[6]
        miner_current_tier = row[7]
        miner_current_scoring_window = row[8]
        miner_current_composite_score = row[9]
        miner_current_sharpe_ratio = row[10]
        miner_current_sortino_ratio = row[11]
        miner_current_roi = row[12]
        miner_current_clv_avg = row[13]
        miner_last_prediction_date = row[14]
        miner_lifetime_earnings = row[15]
        miner_lifetime_wager_amount = row[16]
        miner_lifetime_roi  = row[17]
        miner_lifetime_predictions = row[18]
        miner_lifetime_wins = row[19]
        miner_lifetime_losses = row[20]
        miner_win_loss_ratio = row[21]
        return cls(
            miner_hotkey=miner_hotkey,
            miner_coldkey=miner_coldkey,
            miner_uid=miner_uid,
            miner_rank=miner_rank,
            miner_status=miner_status,
            miner_cash=miner_cash,
            miner_current_incentive=miner_current_incentive,
            miner_current_tier=miner_current_tier,
            miner_current_scoring_window=miner_current_scoring_window,
            miner_current_composite_score=miner_current_composite_score,
            miner_current_sharpe_ratio=miner_current_sharpe_ratio,
            miner_current_sortino_ratio=miner_current_sortino_ratio,
            miner_current_roi=miner_current_roi,
            miner_current_clv_avg=miner_current_clv_avg,
            miner_last_prediction_date=miner_last_prediction_date,
            miner_lifetime_earnings=miner_lifetime_earnings,
            miner_lifetime_wager_amount=miner_lifetime_wager_amount,
            miner_lifetime_roi=miner_lifetime_roi,
            miner_lifetime_predictions=miner_lifetime_predictions,
            miner_lifetime_wins=miner_lifetime_wins,
            miner_lifetime_losses=miner_lifetime_losses,
            miner_win_loss_ratio=miner_win_loss_ratio,
        )


class Metadata(BaseModel):
    """Synapse Metadata class, add more fields if needed"""

    synapse_id: str = Field(..., description="UUID of the synapse")
    neuron_uid: str = Field(..., description="UUID of the serving neuron")
    timestamp: str = Field(..., description="Timestamp of the synapse")
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
            subnet_id: str
        Returns:
            Metadata: A new metadata object to attach to a synapse
        """
        synapse_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        bt.logging.debug(
            f"Creating Metadata with synapse_id: {synapse_id}, neuron_uid: {neuron_uid}, timestamp: {timestamp}, subnet_version: {subnet_version}"
        )
        return Metadata(
            synapse_id=synapse_id,
            neuron_uid=neuron_uid,
            timestamp=timestamp,
            subnet_version=subnet_version,
            synapse_type=synapse_type,
        )


class TeamGamePrediction(BaseModel):
    """
    Data class from json. Will need to be modified in the future for more complex prediction types.
    """

    prediction_id: str = Field(..., description="UUID of the prediction")
    game_id: str = Field(..., description="Game ID - Not Unique (External ID from API)")
    miner_uid: str = Field(
        ..., description="UUID or UID of the miner that made the prediction"
    )
    prediction_date: str = Field(..., description="Prediction date of the prediction")
    predicted_outcome: str = Field(..., description="Predicted outcome")
    predicted_odds: float = Field(
        ..., description="Predicted outcome odds at the time of prediction"
    )
    team_a: Optional[str] = Field(None, description="Team A, typically the home team")
    team_b: Optional[str] = Field(None, description="Team B, typically the away team")
    wager: float = Field(..., description="Wager of the prediction")
    team_a_odds: float = Field(..., description="Team A odds at the time of prediction")
    team_b_odds: float = Field(..., description="Team B odds at the time of prediction")
    tie_odds: Optional[float] = Field(
        None, description="Tie odds at the time of prediction"
    )
    model_name: Optional[str] = Field(
        None,
        description="Name of the model that made the prediction - null if submitted by a human",
    )
    confidence_score: Optional[float] = Field(
        None, description="Confidence score of the prediction (model based predictions only)"
    )
    outcome: str = Field(..., description="Outcome of prediction")
    payout: float = Field(
        ...,
        description="Payout of prediction - for local tracking, not used in scoring",
    )


class TeamGame(BaseModel):
    """
    Data class from json. May need to be modified in the future for more complex prediction types
    """

    game_id: str = Field(
        ...,
        description="ID of the team game - Formerly 'externalId' (No need for unique ID here)",
    )
    team_a: str = Field(..., description="Team A (Typically the home team)")
    team_b: str = Field(..., description="Team B (Typically the away team)")
    sport: str = Field(..., description="Sport")
    league: str = Field(..., description="League name")
    create_date: str = Field(..., description="Create date")
    last_update_date: str = Field(..., description="Last update date")
    event_start_date: str = Field(..., description="Event start date")
    active: bool = Field(..., description="Active")
    outcome: str = Field(..., description="Outcome")
    team_a_odds: float = Field(..., description="Team A odds")
    team_b_odds: float = Field(..., description="Team B odds")
    tie_odds: float = Field(..., description="Tie odds")
    can_tie: bool = Field(..., description="Can tie")


class GameData(bt.Synapse):
    """
    This class defines the synapse object for game data, consisting of a dictionary of TeamGame objects with a UUID as key.
    """

    metadata: Optional[Metadata]
    gamedata_dict: Optional[Dict[str, TeamGame]]
    prediction_dict: Optional[Dict[str, TeamGamePrediction]]
    confirmation_dict: Optional[Dict[str, TeamGamePrediction]]
    error: Optional[str]

    @classmethod
    def create(
        cls,
        db_path: str,
        wallet: bt.wallet,
        subnet_version: str,
        neuron_uid: int,  # Note: This is an int
        synapse_type: str,
        gamedata_dict: Dict[str, TeamGame] = None,
        prediction_dict: Dict[str, TeamGamePrediction] = None,
        confirmation_dict: Dict[str, TeamGamePrediction] = None,
    ):
        metadata = Metadata.create(
            wallet=wallet,
            subnet_version=subnet_version,
            neuron_uid=str(neuron_uid),  # Convert to string here
            synapse_type=synapse_type,
        )
        if synapse_type == "prediction":
            gamedata_dict = None
        elif synapse_type == "game_data":
            prediction_dict = None

            return cls(
                metadata=metadata,
                gamedata_dict=gamedata_dict,
                prediction_dict=prediction_dict,
                synapse_type=synapse_type,
            )

    def deserialize(self):
        return self.gamedata_dict, self.prediction_dict, self.metadata


