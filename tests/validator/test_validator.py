import unittest
from unittest.mock import Mock, patch, MagicMock, call
import torch
import bittensor as bt
import sys
import os
from datetime import datetime, timezone, timedelta
import sqlite3
import uuid

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bettensor.validator.bettensor_validator import BettensorValidator
from bettensor.protocol import GameData, Metadata, TeamGamePrediction

class TestBettensorValidator(unittest.TestCase):

    def setUp(self):
        # Mock the ArgumentParser
        self.mock_parser = Mock()
        self.mock_parser._actions = []
        
        # Add mock arguments
        mock_args = [
            Mock(dest='subtensor.network'),
            Mock(dest='subtensor.chain_endpoint'),
            Mock(dest='wallet.name'),
            Mock(dest='wallet.hotkey'),
            Mock(dest='logging.trace'),
            Mock(dest='logging.debug'),
            Mock(dest='logging.info'),
            Mock(dest='use_bt_api'),
            Mock(dest='netuid'),
            Mock(dest='axon.port'),
        ]
        self.mock_parser._actions.extend(mock_args)
        
        # Mock the parse_args method
        self.mock_parser.parse_args.return_value = Mock(
            subtensor=Mock(network='mocknet', chain_endpoint='mock_endpoint'),
            wallet=Mock(name='mock_wallet', hotkey='mock_hotkey'),
            logging=Mock(trace=False, debug=False, info=True),
            use_bt_api=False,
            netuid=1,
            axon=Mock(port=8080)
        )

        self.validator = BettensorValidator(self.mock_parser)
        
        # Mock necessary attributes and methods
        self.validator.subtensor = Mock()
        self.validator.wallet = Mock()
        self.validator.metagraph = Mock()
        self.validator.dendrite = Mock()
        self.validator.sports_data = Mock()
        self.validator.weight_setter = Mock()
        self.validator.db_path = ':memory:'  # Use in-memory SQLite database for testing
        self.validator.max_targets = 256
        self.validator.neuron_config = Mock(netuid=1)

        # Create the game_data table
        self.conn = sqlite3.connect(self.validator.db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_data (
                id TEXT PRIMARY KEY,
                sport TEXT,
                league TEXT,
                event_start_date TEXT,
                team_a TEXT,
                team_b TEXT,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                outcome TEXT,
                external_id TEXT,
                active INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                game_id TEXT,
                miner_uid TEXT,
                prediction_date TEXT,
                predicted_outcome TEXT,
                predicted_odds REAL,
                team_a TEXT,
                team_b TEXT,
                wager REAL,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                is_model_prediction BOOLEAN,
                outcome TEXT,
                payout REAL,
                sent_to_site INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_initialize_connection(self):
        with patch('bittensor.subtensor') as mock_subtensor:
            mock_subtensor_instance = Mock()
            mock_subtensor.return_value = mock_subtensor_instance
            result = self.validator.initialize_connection()
            self.assertIsNotNone(result)
            mock_subtensor.assert_called_once_with(config=self.validator.neuron_config)

    def test_sync_metagraph(self):
        mock_subtensor = Mock()
        self.validator.subtensor = mock_subtensor
        mock_metagraph = Mock()
        self.validator.metagraph = mock_metagraph
        
        result = self.validator.sync_metagraph()
        
        mock_metagraph.sync.assert_called_once_with(subtensor=mock_subtensor, lite=True)
        self.assertEqual(result, mock_metagraph)

    @patch('bittensor.axon')
    def test_serve_axon(self, mock_axon):
        mock_axon_instance = Mock()
        mock_axon.return_value = mock_axon_instance
        
        self.validator.serve_axon()
        
        mock_axon.assert_called_once_with(wallet=self.validator.wallet)
        mock_axon_instance.serve.assert_called_once_with(
            netuid=self.validator.neuron_config.netuid, 
            subtensor=self.validator.subtensor
        )

    def test_get_uids_to_query(self):
        # Mock necessary attributes
        self.validator.metagraph.total_stake = torch.tensor([1.0, 0.0, 2.0])
        self.validator.metagraph.neurons = [
            Mock(axon_info=Mock(ip='1.1.1.1')),
            Mock(axon_info=Mock(ip='0.0.0.0')),
            Mock(axon_info=Mock(ip='2.2.2.2'))
        ]
        self.validator.metagraph.uids = torch.tensor([0, 1, 2])
        self.validator.metagraph.hotkeys = ['hotkey1', 'hotkey2', 'hotkey3']
        self.validator.blacklisted_miner_hotkeys = ['hotkey2']
        
        all_axons = [Mock(hotkey='hotkey1'), Mock(hotkey='hotkey2'), Mock(hotkey='hotkey3')]
        
        uids_to_query, list_of_uids, blacklisted_uids, uids_not_to_query = self.validator.get_uids_to_query(all_axons)
        
        self.assertEqual(len(uids_to_query), 2)
        self.assertEqual(list_of_uids, [0, 2])
        self.assertEqual(blacklisted_uids, [1])
        self.assertEqual(uids_not_to_query, [1])

    @patch('sqlite3.connect')
    def test_process_prediction(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchone.side_effect = [
            (0,),  # First call for prediction count
            ('soccer', 'Premier League', '2023-09-16T15:00:00', 'Team A', 'Team B', 2.0, 3.0, 3.5, 'Unfinished'),  # Second call for game data
            (0,),  # Third call for prediction count
            ('baseball', 'MLB', '2023-09-16T19:00:00', 'Team C', 'Team D', 1.8, 2.2, None, 'Unfinished'),  # Fourth call for game data
        ]

        processed_uids = [0, 1]
        predictions = [
            [
                {'game_1': GameData()},
                {'pred_id_1': TeamGamePrediction(
                    prediction_id='pred_id_1',
                    game_id='game_1',
                    miner_uid='0',
                    prediction_date='2023-01-01',
                    predicted_outcome='Team A',
                    predicted_odds=1.5,
                    wager=100,
                    team_a_odds=1.5,
                    team_b_odds=2.5,
                    tie_odds=3.0,
                    outcome='Unfinished',
                    payout=0
                )},
                Metadata(
                    neuron_uid=0,
                    synapse_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    signature="mock_signature",
                    subnet_version="v1.0.0",
                    synapse_type="game_data"
                )
            ],
            [
                {'game_2': GameData()},
                {'pred_id_2': TeamGamePrediction(
                    prediction_id='pred_id_2',
                    game_id='game_2',
                    miner_uid='1',
                    prediction_date='2023-01-01',
                    predicted_outcome='Team B',
                    predicted_odds=2.0,
                    wager=200,
                    team_a_odds=2.0,
                    team_b_odds=1.8,
                    tie_odds=3.5,
                    outcome='Unfinished',
                    payout=0
                )},
                Metadata(
                    neuron_uid=1,
                    synapse_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    signature="mock_signature",
                    subnet_version="v1.0.0",
                    synapse_type="game_data"
                )
            ]
        ]

        self.validator.metagraph.hotkeys = ['hotkey1', 'hotkey2']

        self.validator.process_prediction(processed_uids, predictions)

        # Assert that the database connection was established (it's okay if it's called twice due to setup/teardown)
        self.assertGreaterEqual(mock_connect.call_count, 1)

        # Assert that the correct SQL queries were executed
        expected_calls = [
            call("SELECT COUNT(*) FROM predictions WHERE prediction_id = ?", ('pred_id_1',)),
            call("SELECT sport, league, event_start_date, team_a, team_b, team_a_odds, team_b_odds, tie_odds, outcome FROM game_data WHERE id = ?", ('game_1',)),
            call("SELECT COUNT(*) FROM predictions WHERE prediction_id = ?", ('pred_id_2',)),
            call("SELECT sport, league, event_start_date, team_a, team_b, team_a_odds, team_b_odds, tie_odds, outcome FROM game_data WHERE id = ?", ('game_2',)),
        ]
        mock_cursor.execute.assert_has_calls(expected_calls, any_order=True)

    @patch('sqlite3.connect')
    def test_update_game_outcome(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value

        game_id = 'game_1'
        numeric_outcome = 0

        self.validator.update_game_outcome(game_id, numeric_outcome)

        mock_connect.assert_called_once_with(self.validator.db_path)
        mock_cursor.execute.assert_called_once_with(
            "UPDATE game_data SET outcome = ?, active = 0 WHERE externalId = ?",
            (numeric_outcome, game_id)
        )
        mock_conn.commit.assert_called_once()

    @patch('sqlite3.connect')
    def test_get_recent_games(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value

        mock_cursor.fetchall.return_value = [
            ('game_1', 'Team A', 'Team B', 'ext_1'),
            ('game_2', 'Team C', 'Team D', 'ext_2')
        ]

        result = self.validator.get_recent_games()

        mock_connect.assert_called_once_with(self.validator.db_path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ('game_1', 'Team A', 'Team B', 'ext_1'))
        self.assertEqual(result[1], ('game_2', 'Team C', 'Team D', 'ext_2'))

    @patch.object(BettensorValidator, 'get_sport_from_db')
    @patch.object(BettensorValidator, 'process_game_result')
    def test_determine_winner(self, mock_process_game_result, mock_get_sport_from_db):
        mock_get_sport_from_db.return_value = 'soccer'
        self.validator.api_client = Mock()
        mock_game_data = {
            'response': [{
                'fixture': {'status': {'long': 'Match Finished'}},
                'goals': {'home': 2, 'away': 1}
            }]
        }
        self.validator.api_client.get_soccer_game.return_value = mock_game_data

        game_info = ('game_1', 'Team A', 'Team B', 'ext_1')
        self.validator.determine_winner(game_info)

        mock_get_sport_from_db.assert_called_once_with('ext_1')
        self.validator.api_client.get_soccer_game.assert_called_once_with('ext_1')
        mock_process_game_result.assert_called_once_with('soccer', mock_game_data['response'][0], 'ext_1', 'Team A', 'Team B')

        # Add assertions to check if the correct methods were called
        self.assertTrue(self.validator.api_client.get_soccer_game.called)
        self.assertFalse(self.validator.api_client.get_baseball_game.called)
        self.assertFalse(self.validator.api_client.get_nfl_result.called)

    def test_process_game_result(self):
        with patch.object(BettensorValidator, 'update_game_outcome') as mock_update_game_outcome:
            sport = 'soccer'
            game_response = {
                'goals': {'home': 2, 'away': 1}
            }
            external_id = 'ext_1'
            team_a = 'Team A'
            team_b = 'Team B'

            self.validator.process_game_result(sport, game_response, external_id, team_a, team_b)

            mock_update_game_outcome.assert_called_once_with('ext_1', 0)

    @patch('sqlite3.connect')
    def test_update_recent_games(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            ('game_1', 'Team A', 'Team B', 'ext_1', '2023-06-01T12:00:00+00:00', 'soccer'),
            ('game_2', 'Team C', 'Team D', 'ext_2', '2023-06-01T13:00:00+00:00', 'baseball')
        ]

        with patch.object(BettensorValidator, 'determine_winner') as mock_determine_winner:
            self.validator.update_recent_games()

            mock_connect.assert_called_once_with(self.validator.db_path)
            mock_cursor.execute.assert_called_once()
            mock_determine_winner.assert_has_calls([
                call(('game_1', 'Team A', 'Team B', 'ext_1')),
                call(('game_2', 'Team C', 'Team D', 'ext_2'))
            ])

    def test_set_weights(self):
        self.validator.weight_setter.set_weights.return_value = True

        result = self.validator.set_weights()

        self.validator.weight_setter.set_weights.assert_called_once_with(self.validator.db_path)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()