"""
test script for protocol functions and files
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from bettensor.protocol import MinerStats, Metadata, TeamGamePrediction, TeamGame, GameData
import bittensor as bt
import uuid

@pytest.fixture
def mock_wallet():
    return MagicMock(spec=bt.wallet)

def test_miner_stats_create():
    row = ("hotkey", "coldkey", "uid", 1, 1000.0, 10.0, "2023-05-01", 5000.0, 2000.0, 100, 60, 40, 0.6, "active")
    miner_stats = MinerStats.create(row)
    
    assert miner_stats.miner_hotkey == "hotkey"
    assert miner_stats.miner_coldkey == "coldkey"
    assert miner_stats.miner_uid == "uid"
    assert miner_stats.miner_rank == 1
    assert miner_stats.miner_cash == 1000.0
    assert miner_stats.miner_current_incentive == 10.0
    assert miner_stats.miner_last_prediction_date == "2023-05-01"
    assert miner_stats.miner_lifetime_earnings == 5000.0
    assert miner_stats.miner_lifetime_wager == 2000.0
    assert miner_stats.miner_lifetime_predictions == 100
    assert miner_stats.miner_lifetime_wins == 60
    assert miner_stats.miner_lifetime_losses == 40
    assert miner_stats.miner_win_loss_ratio == 0.6
    assert miner_stats.miner_status == "active"

def test_metadata_create(mock_wallet):
    with patch('bettensor.protocol.create_signature', return_value='test_signature'):
        with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
            metadata = Metadata.create(mock_wallet, "1.0", "test_uid", "prediction")
            
            assert metadata.synapse_id == "12345678-1234-5678-1234-567812345678"
            assert metadata.neuron_uid == "test_uid"
            assert metadata.signature == "test_signature"
            assert metadata.subnet_version == "1.0"
            assert metadata.synapse_type == "prediction"

def test_team_game_prediction():
    prediction = TeamGamePrediction(
        predictionID="pred1",
        teamGameID="game1",
        minerID="miner1",
        predictionDate="2023-05-01",
        predictedOutcome="TeamA",
        wager=100.0,
        teamAodds=1.5,
        teamBodds=2.5,
        tieOdds=3.0,
        outcome="Win",
        can_overwrite=True
    )
    
    assert prediction.predictionID == "pred1"
    assert prediction.teamGameID == "game1"
    assert prediction.minerID == "miner1"
    assert prediction.predictionDate == "2023-05-01"
    assert prediction.predictedOutcome == "TeamA"
    assert prediction.wager == 100.0
    assert prediction.teamAodds == 1.5
    assert prediction.teamBodds == 2.5
    assert prediction.tieOdds == 3.0
    assert prediction.outcome == "Win"
    assert prediction.can_overwrite == True

def test_team_game():
    game = TeamGame(
        id="game1",
        teamA="TeamA",
        teamB="TeamB",
        sport="Football",
        league="NFL",
        externalId="ext1",
        createDate="2023-05-01",
        lastUpdateDate="2023-05-02",
        eventStartDate="2023-05-10",
        active=True,
        outcome="Pending",
        teamAodds=1.5,
        teamBodds=2.5,
        tieOdds=3.0,
        canTie=True
    )
    
    assert game.id == "game1"
    assert game.teamA == "TeamA"
    assert game.teamB == "TeamB"
    assert game.sport == "Football"
    assert game.league == "NFL"
    assert game.externalId == "ext1"
    assert game.createDate == "2023-05-01"
    assert game.lastUpdateDate == "2023-05-02"
    assert game.eventStartDate == "2023-05-10"
    assert game.active == True
    assert game.outcome == "Pending"
    assert game.teamAodds == 1.5
    assert game.teamBodds == 2.5
    assert game.tieOdds == 3.0
    assert game.canTie == True

@patch('bettensor.protocol.sqlite3.connect')
def test_game_data_create(mock_connect, mock_wallet):
    mock_cursor = MagicMock()
    mock_connect.return_value.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        ("game1", "TeamA", "TeamB", "Football", "NFL", "ext1", "2023-05-01", "2023-05-02", "2023-05-10", 1, "Pending", 1.5, 2.5, 3.0, 1)
    ]
    
    with patch('bettensor.protocol.Metadata.create') as mock_metadata_create:
        mock_metadata = MagicMock()
        mock_metadata.timestamp = "2023-05-01T00:00:00+00:00"
        mock_metadata_create.return_value = mock_metadata
        
        game_data = GameData.create("test.db", mock_wallet, "1.0", "test_uid", "game_data")
        
        assert isinstance(game_data.metadata, Metadata)
        assert len(game_data.gamedata_dict) == 1
        assert "game1" in game_data.gamedata_dict
        assert isinstance(game_data.gamedata_dict["game1"], TeamGame)

def test_game_data_deserialize():
    metadata = MagicMock(spec=Metadata)
    gamedata_dict = {"game1": MagicMock(spec=TeamGame)}
    prediction_dict = {"pred1": MagicMock(spec=TeamGamePrediction)}
    
    game_data = GameData(metadata=metadata, gamedata_dict=gamedata_dict, prediction_dict=prediction_dict)
    
    deserialized_gamedata, deserialized_predictions, deserialized_metadata = game_data.deserialize()
    
    assert deserialized_gamedata == gamedata_dict
    assert deserialized_predictions == prediction_dict
    assert deserialized_metadata == metadata