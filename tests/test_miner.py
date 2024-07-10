"""
test script for miner functions and files
"""

import pytest
from unittest.mock import MagicMock, patch
from bettensor.miner.bettensor_miner import BettensorMiner
from bettensor.protocol import GameData, Metadata
from bettensor.utils.miner_stats import MinerStatsHandler
from bettensor.protocol import MinerStats

@pytest.fixture
def mock_argparser():
    parser = MagicMock()
    parser.parse_args.return_value = MagicMock(
        db_path="./test_data/miner.db",
        miner_set_weights="False",
        validator_min_stake=1000.0
    )
    return parser

@pytest.fixture
def mock_bettensor_miner(mock_argparser):
    with patch("bettensor.miner.bettensor_miner.bt") as mock_bt:
        mock_bt.wallet.return_value = MagicMock(hotkey=MagicMock(ss58_address="test_hotkey"))
        mock_bt.subtensor.return_value = MagicMock()
        mock_bt.metagraph.return_value = MagicMock(hotkeys=["test_hotkey"])
        yield BettensorMiner(parser=mock_argparser)

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
        metadata=Metadata(subnet_version=1),
        gamedata_dict={"game1": MagicMock()}
    )
    with patch.object(mock_bettensor_miner, "add_game_data") as mock_add_game_data:
        result = mock_bettensor_miner.forward(mock_synapse)
        mock_add_game_data.assert_called_once_with({"game1": MagicMock()})
        assert isinstance(result, GameData)

import pytest
from unittest.mock import MagicMock, patch
from neurons.miner import main

@pytest.fixture
def mock_bettensor_miner():
    return MagicMock()

@patch("neurons.miner.bt")
def test_main(mock_bt, mock_bettensor_miner):
    mock_axon = MagicMock()
    mock_bt.axon.return_value = mock_axon

    main(mock_bettensor_miner)

    mock_axon.attach.assert_called_once()
    mock_axon.serve.assert_called_once()
    mock_axon.start.assert_called_once()

    # Test the main loop
    mock_bettensor_miner.step = 0
    mock_bettensor_miner.subtensor.block = 100
    mock_bettensor_miner.last_updated_block = 0

    with patch("neurons.miner.time.sleep", side_effect=KeyboardInterrupt):
        main(mock_bettensor_miner)

    assert mock_bettensor_miner.metagraph.sync.called
    assert mock_bt.logging.info.called



@pytest.fixture
def mock_miner():
    return MagicMock(db_manager=MagicMock())

@pytest.fixture
def miner_stats_handler(mock_miner):
    return MinerStatsHandler(mock_miner)

def test_create_table(miner_stats_handler):
    with patch.object(miner_stats_handler.db_manager, "get_cursor") as mock_get_cursor:
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        miner_stats_handler.create_table()

        mock_cursor.execute.assert_called_once()
        mock_cursor.connection.commit.assert_called_once()

def test_update_miner_row(miner_stats_handler):
    mock_stats = MagicMock(spec=MinerStats)
    with patch.object(miner_stats_handler.db_manager, "get_cursor") as mock_get_cursor:
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        miner_stats_handler.update_miner_row(mock_stats)

        mock_cursor.execute.assert_called_once()
        mock_cursor.connection.commit.assert_called_once()

def test_reset_daily_cash_timed(miner_stats_handler):
    with patch.object(miner_stats_handler.db_manager, "get_cursor") as mock_get_cursor:
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        miner_stats_handler.reset_daily_cash_timed()

        mock_cursor.execute.assert_called_once_with(
            """
            UPDATE miner_stats
            SET miner_cash = 1000
            """
        )
        mock_cursor.connection.commit.assert_called_once()

def test_init_miner_row(miner_stats_handler):
    with patch.object(miner_stats_handler.db_manager, "get_cursor") as mock_get_cursor:
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        result = miner_stats_handler.init_miner_row("test_hotkey", 1)

        assert result == True
        assert mock_cursor.execute.call_count == 2
        mock_cursor.connection.commit.assert_called_once()

def test_return_miner_stats(miner_stats_handler):
    with patch.object(miner_stats_handler.db_manager, "get_cursor") as mock_get_cursor:
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (
            "test_hotkey", "test_coldkey", 1, 0, 1000, 0, "", 0, 0, 0, 0, 0, 0, "active"
        )

        result = miner_stats_handler.return_miner_stats("test_hotkey")

        assert isinstance(result, MinerStats)
        assert result.miner_hotkey == "test_hotkey"
        assert result.miner_uid == 1
        assert result.miner_cash == 1000