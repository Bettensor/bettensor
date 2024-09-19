"""
test script for miner functions and files
"""

import pytest
from unittest.mock import MagicMock, patch
import datetime
import pytz
import torch
from bettensor.miner.bettensor_miner import BettensorMiner
from bettensor.protocol import GameData, Metadata, TeamGame, TeamGamePrediction
from bettensor.miner.stats.miner_stats import MinerStateManager, MinerStatsHandler
from bettensor.miner.database.database_manager import DatabaseManager
from bettensor.miner.database.games import GamesHandler
from bettensor.miner.database.predictions import PredictionsHandler


@pytest.fixture
def mock_argparser():
    parser = MagicMock()
    parser.parse_args.return_value = MagicMock(
        db_path="./test_data/miner.db",
        miner_set_weights="False",
        validator_min_stake=1000.0,
    )
    return parser


@pytest.fixture
def mock_bettensor_miner(mock_argparser):
    with patch("bettensor.miner.bettensor_miner.bt") as mock_bt:
        mock_bt.wallet.return_value = MagicMock(
            hotkey=MagicMock(ss58_address="test_hotkey")
        )
        mock_bt.subtensor.return_value = MagicMock()
        mock_bt.metagraph.return_value = MagicMock(hotkeys=["test_hotkey"])
        yield BettensorMiner(parser=mock_argparser)


# Tests for BettensorMiner
def test_bettensor_miner_initialization(mock_bettensor_miner):
    assert mock_bettensor_miner.db_path == "./test_data/miner.db"
    assert mock_bettensor_miner.miner_set_weights == False
    assert mock_bettensor_miner.validator_min_stake == 1000.0


def test_blacklist(mock_bettensor_miner):
    mock_synapse = MagicMock(dendrite=MagicMock(hotkey="test_hotkey"))
    mock_bettensor_miner.metagraph.hotkeys = ["test_hotkey"]
    mock_bettensor_miner.metagraph.validator_permit = [True]
    mock_bettensor_miner.metagraph.S = [2000.0]

    result, reason = mock_bettensor_miner.blacklist(mock_synapse)
    assert result == False
    assert "Accepted hotkey" in reason


def test_priority(mock_bettensor_miner):
    mock_synapse = MagicMock(dendrite=MagicMock(hotkey="test_hotkey"))
    mock_bettensor_miner.metagraph.hotkeys = ["test_hotkey"]
    mock_bettensor_miner.metagraph.S = [2000.0]

    priority = mock_bettensor_miner.priority(mock_synapse)
    assert priority == 2000.0


def test_forward(mock_bettensor_miner):
    mock_synapse = MagicMock(
        metadata=Metadata(subnet_version=1), gamedata_dict={"game1": MagicMock()}
    )
    with patch.object(
        mock_bettensor_miner.games_handler, "process_games"
    ) as mock_process_games, patch.object(
        mock_bettensor_miner.predictions_handler, "process_predictions"
    ) as mock_process_predictions:
        mock_process_games.return_value = ({}, {})
        mock_process_predictions.return_value = {"pred1": MagicMock()}
        result = mock_bettensor_miner.forward(mock_synapse)
        assert isinstance(result, GameData)
        assert result.prediction_dict == {"pred1": MagicMock()}


# Tests for GamesHandler
@pytest.fixture
def mock_games_handler():
    return GamesHandler(MagicMock())


def test_process_games(mock_games_handler):
    game_data_dict = {
        "game1": TeamGame(id="game1", teamA="TeamA", teamB="TeamB", externalId="ext1")
    }
    with patch.object(mock_games_handler, "_add_or_update_game") as mock_add_or_update:
        mock_add_or_update.return_value = True
        updated, new = mock_games_handler.process_games(game_data_dict)
        assert len(new) == 1
        assert "game1" in new


def test_get_active_games(mock_games_handler):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        (
            "game1",
            "TeamA",
            1.5,
            "TeamB",
            2.5,
            "Sport",
            "League",
            "ext1",
            "2023-01-01",
            "2023-01-01",
            "2023-01-02",
            1,
            None,
            3.0,
            True,
        )
    ]
    mock_games_handler.db_manager.get_cursor.return_value.__enter__.return_value = (
        mock_cursor
    )

    active_games = mock_games_handler.get_active_games()
    assert len(active_games) == 1
    assert isinstance(active_games["game1"], TeamGame)


# Tests for PredictionsHandler
@pytest.fixture
def mock_predictions_handler():
    return PredictionsHandler(MagicMock(), MagicMock(), "test_miner")


def test_process_predictions(mock_predictions_handler):
    updated_games = {
        "game1": TeamGame(id="game1", teamA="TeamA", teamB="TeamB", externalId="ext1")
    }
    new_games = {
        "game2": TeamGame(id="game2", teamA="TeamC", teamB="TeamD", externalId="ext2")
    }

    with patch.object(
        mock_predictions_handler, "_update_prediction_outcome"
    ) as mock_update, patch.object(
        mock_predictions_handler, "_get_or_create_prediction"
    ) as mock_get_or_create:
        mock_get_or_create.return_value = TeamGamePrediction(
            predictionID="pred1", teamGameID="game2"
        )
        result = mock_predictions_handler.process_predictions(updated_games, new_games)
        assert len(result) == 1
        assert "pred1" in result


# Tests for DatabaseManager
@pytest.fixture
def mock_db_manager():
    with patch("bettensor.miner.database.database_manager.sqlite3") as mock_sqlite:
        yield DatabaseManager("test.db")


def test_get_cursor(mock_db_manager):
    with mock_db_manager.get_cursor() as cursor:
        assert cursor is not None


def test_ensure_db_directory_exists(mock_db_manager):
    with patch("os.path.exists") as mock_exists, patch("os.makedirs") as mock_makedirs:
        mock_exists.return_value = False
        mock_db_manager.ensure_db_directory_exists()
        mock_makedirs.assert_called_once()


# Tests for MinerStateManager
@pytest.fixture
def mock_state_manager():
    return MinerStateManager(MagicMock(), "test_hotkey", "test_uid")


def test_load_state(mock_state_manager):
    with patch.object(
        mock_state_manager, "_load_state_from_file"
    ) as mock_file_load, patch.object(
        mock_state_manager, "_load_state_from_db"
    ) as mock_db_load, patch.object(
        mock_state_manager, "_get_initial_state"
    ) as mock_initial_state:
        mock_file_load.return_value = None
        mock_db_load.return_value = None
        mock_initial_state.return_value = {"miner_cash": torch.tensor(1000)}
        mock_state_manager.load_state()
        assert mock_state_manager.state["miner_cash"] == 1000


def test_update_on_prediction(mock_state_manager):
    mock_state_manager.state = {
        "miner_lifetime_predictions": torch.tensor(0),
        "miner_cash": torch.tensor(1000),
    }
    mock_state_manager.update_on_prediction({"wager": 100})
    assert mock_state_manager.state["miner_lifetime_predictions"] == 1
    assert mock_state_manager.state["miner_cash"] == 900


# Tests for MinerStatsHandler
@pytest.fixture
def mock_stats_handler():
    return MinerStatsHandler(MagicMock())


def test_init_miner_row(mock_stats_handler):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_stats_handler.db_manager.get_cursor.return_value.__enter__.return_value = (
        mock_cursor
    )

    mock_stats_handler.init_miner_row()
    mock_cursor.execute.assert_called()


def test_update_miner_row(mock_stats_handler):
    mock_cursor = MagicMock()
    mock_stats_handler.db_manager.get_cursor.return_value.__enter__.return_value = (
        mock_cursor
    )

    stats = {
        "miner_hotkey": "test_hotkey",
        "miner_cash": 1000,
        "miner_lifetime_earnings": 500,
        "miner_lifetime_predictions": 10,
        "miner_lifetime_wins": 5,
        "miner_lifetime_losses": 5,
        "miner_win_loss_ratio": 0.5,
        "miner_last_prediction_date": "2023-01-01",
    }
    mock_stats_handler.update_miner_row(stats)
    mock_cursor.execute.assert_called()


# Add more tests as needed for other methods and edge cases
