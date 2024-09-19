"""
test script for validator functions and files
"""

import pytest
from unittest.mock import MagicMock, patch
from neurons.validator import main
from bettensor.validator.bettensor_validator import BettensorValidator
import torch


@pytest.fixture
def mock_validator():
    return MagicMock(spec=BettensorValidator)


@pytest.fixture
def mock_argparser():
    parser = MagicMock()
    parser.parse_args.return_value = MagicMock(
        db="test.db", alpha=0.9, netuid=30, max_targets=128, load_state="True"
    )
    return parser


@pytest.fixture
def validator(mock_argparser):
    with patch("bettensor.validator.bettensor_validator.bt") as mock_bt:
        mock_bt.wallet.return_value = MagicMock(
            hotkey=MagicMock(ss58_address="test_hotkey")
        )
        mock_bt.subtensor.return_value = MagicMock()
        mock_bt.metagraph.return_value = MagicMock(hotkeys=["test_hotkey"])
        yield BettensorValidator(parser=mock_argparser)


# Tests for validator.py


@patch("neurons.validator.SportsData")
@patch("neurons.validator.load_dotenv")
@patch("neurons.validator.os.getenv")
def test_main(mock_getenv, mock_load_dotenv, mock_sports_data, mock_validator):
    mock_getenv.return_value = "fake_api_key"
    mock_sports_data.return_value.get_multiple_game_data.return_value = {}

    with patch("neurons.validator.time.sleep", side_effect=KeyboardInterrupt):
        main(mock_validator)

    mock_load_dotenv.assert_called_once()
    mock_getenv.assert_called_once_with("RAPID_API_KEY")
    mock_sports_data.assert_called_once()
    mock_sports_data.return_value.get_multiple_game_data.assert_called()

    mock_validator.sync_metagraph.assert_called()
    mock_validator.check_hotkeys.assert_called()
    mock_validator.save_state.assert_called()
    mock_validator.set_weights.assert_called()


@pytest.mark.parametrize(
    "step,expected_calls",
    [
        (0, 0),
        (5, 1),
        (10, 2),
        (149, 29),
        (150, 30),
    ],
)
def test_main_periodic_actions(step, expected_calls, mock_validator):
    mock_validator.step = step

    with patch("neurons.validator.time.sleep", side_effect=KeyboardInterrupt):
        main(mock_validator)

    assert mock_validator.sync_metagraph.call_count == expected_calls
    assert mock_validator.check_hotkeys.call_count == expected_calls
    assert mock_validator.save_state.call_count == expected_calls


def test_main_error_handling(mock_validator):
    mock_validator.sync_metagraph.side_effect = TimeoutError("Sync failed")

    with patch("neurons.validator.time.sleep", side_effect=KeyboardInterrupt):
        main(mock_validator)

    mock_validator.sync_metagraph.assert_called()
    # Ensure the loop continues despite the error


# Tests for bettensor_validator.py


def test_apply_config(validator):
    bt_classes = [MagicMock(), MagicMock()]
    assert validator.apply_config(bt_classes)
    assert validator.neuron_config is not None


def test_check_vali_reg(validator):
    mock_metagraph = MagicMock(hotkeys=["test_hotkey"])
    mock_wallet = MagicMock(hotkey=MagicMock(ss58_address="test_hotkey"))
    mock_subtensor = MagicMock()

    assert validator.check_vali_reg(mock_metagraph, mock_wallet, mock_subtensor)

    mock_metagraph.hotkeys = ["other_hotkey"]
    assert not validator.check_vali_reg(mock_metagraph, mock_wallet, mock_subtensor)


def test_calculate_total_wager(validator):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(10,), (20,), (30,)]

    total_wager = validator.calculate_total_wager(mock_cursor, "miner1", "2023-05-01")
    assert total_wager == 60

    total_wager = validator.calculate_total_wager(
        mock_cursor, "miner1", "2023-05-01", exclude_id="game1"
    )
    assert total_wager == 60
    mock_cursor.execute.assert_called_with(
        "SELECT p.wager FROM predictions p JOIN game_data g ON p.teamGameId = g.id WHERE p.minerId = ? AND DATE(g.eventStartDate) = DATE(?) AND p.teamGameId != ?",
        ("miner1", "2023-05-01", "game1"),
    )


def test_check_hotkeys(validator):
    validator.metagraph = MagicMock(hotkeys=["hotkey1", "hotkey2"])
    validator.hotkeys = ["hotkey1", "hotkey2"]
    validator.scores = torch.tensor([1.0, 1.0])

    validator.check_hotkeys()
    assert validator.hotkeys == ["hotkey1", "hotkey2"]
    assert torch.all(validator.scores == torch.tensor([1.0, 1.0]))

    validator.metagraph.hotkeys = ["hotkey1", "hotkey3"]
    validator.check_hotkeys()
    assert validator.hotkeys == ["hotkey1", "hotkey3"]
    assert torch.all(validator.scores == torch.tensor([1.0, 0.0]))


def test_init_default_scores(validator):
    validator.metagraph = MagicMock(S=torch.tensor([1.0, 2.0, 3.0]))
    validator.init_default_scores()
    assert torch.all(validator.scores == torch.zeros(3))


@patch("bettensor.validator.bettensor_validator.torch.load")
@patch("bettensor.validator.bettensor_validator.path.exists")
def test_load_state(mock_exists, mock_torch_load, validator):
    mock_exists.return_value = True
    mock_torch_load.return_value = {
        "step": 10,
        "scores": torch.tensor([1.0, 2.0]),
        "hotkeys": ["hotkey1", "hotkey2"],
        "last_updated_block": 100,
        "blacklisted_miner_hotkeys": ["blacklisted_hotkey"],
    }

    validator.load_state()

    assert validator.step == 10
    assert torch.all(validator.scores == torch.tensor([1.0, 2.0]))
    assert validator.hotkeys == ["hotkey1", "hotkey2"]
    assert validator.last_updated_block == 100
    assert validator.blacklisted_miner_hotkeys == ["blacklisted_hotkey"]


@patch("bettensor.validator.bettensor_validator.json.loads")
@patch("builtins.open")
@patch("bettensor.validator.bettensor_validator.Path.is_file")
def test_get_local_miner_blacklist(mock_is_file, mock_open, mock_json_loads, validator):
    mock_is_file.return_value = True
    mock_json_loads.return_value = ["blacklisted_hotkey1", "blacklisted_hotkey2"]

    blacklist = validator._get_local_miner_blacklist()

    assert blacklist == ["blacklisted_hotkey1", "blacklisted_hotkey2"]
    mock_open.assert_called_once()
    mock_json_loads.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", "test_validator.py"])
