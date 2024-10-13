import unittest
from unittest.mock import MagicMock, patch
import bittensor
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bettensor.miner.database.predictions import PredictionsHandler
from bettensor.miner.models.model_utils import NFLPredictor, KellyFractionNet
from bettensor.protocol import TeamGame, TeamGamePrediction
from datetime import datetime, timezone
import joblib
import torch
import scipy.sparse


class TestNFLPredictions(unittest.TestCase):
    def setUp(self):
        self.db_manager = MagicMock()
        self.state_manager = MagicMock()
        self.state_manager.miner_uid = "test_miner_uid"
        self.miner_stats_handler = MagicMock()
        self.miner_stats_handler.stats = {
            "miner_cash": 1000.0,
            "miner_lifetime_wins": 0,
            "miner_lifetime_losses": 0,
            "miner_win_loss_ratio": 0.0,
        }
        self.predictions_handler = PredictionsHandler(
            self.db_manager, self.state_manager, "test_hotkey"
        )
        self.predictions_handler.stats_handler = self.miner_stats_handler

        preprocessor_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "bettensor",
            "miner",
            "models",
            "preprocessor.joblib",
        )
        calibrated_model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "bettensor",
            "miner",
            "models",
            "calibrated_sklearn_model.joblib",
        )
        self.real_preprocessor = joblib.load(preprocessor_path)
        self.real_sklearn = joblib.load(calibrated_model_path)

    @patch("bettensor.miner.models.model_utils.NFLPredictor.get_HFmodel")
    @patch("bettensor.miner.models.model_utils.joblib.load")
    def test_process_model_predictions_nfl(self, mock_joblib_load, mock_get_HFmodel):
        mock_db_manager = MagicMock()
        mock_db_manager.get_model_params.return_value = {
            "nfl_model_on": True,
            "soccer_model_on": True,
            "nfl_minimum_wager_amount": 20.0,
            "nfl_max_wager_amount": 1000,
            "fuzzy_match_percentage": 80,
            "nfl_top_n_games": 10,
            "nfl_kelly_fraction_multiplier": 1.0,
            "nfl_edge_threshold": 0.02,
            "nfl_max_bet_percentage": 0.7,
            "wager_distribution_steepness": 1.0,
            "minimum_wager_amount": 10.0,
            "max_wager_amount": 500,
            "top_n_games": 5,
            "kelly_fraction_multiplier": 0.5,
            "edge_threshold": 0.01,
            "max_bet_percentage": 0.5,
        }

        preprocessor_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "bettensor",
            "miner",
            "models",
            "preprocessor.joblib",
        )
        team_averages_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "bettensor",
            "miner",
            "models",
            "team_historical_stats.csv",
        )
        calibrated_model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "bettensor",
            "miner",
            "models",
            "calibrated_sklearn_model.joblib",
        )

        mock_miner_stats_handler = MagicMock()
        mock_miner_stats_handler.get_miner_cash.return_value = 1000
        mock_miner_stats_handler.stats = {
            "miner_lifetime_wins": 10,
            "miner_lifetime_losses": 5,
            "miner_win_loss_ratio": 2.0,
            "miner_cash": 1000.0,
        }
        mock_miner_stats_handler.update_stats_from_predictions.side_effect = (
            lambda x: None
        )

        predictions_handler = PredictionsHandler(
            mock_db_manager, self.state_manager, "test_hotkey"
        )
        predictions_handler.stats_handler = mock_miner_stats_handler

        def real_get_best_match(team_name, encoded_teams, sport):
            return team_name

        predictions_handler.get_best_match = real_get_best_match

        nfl_predictor = NFLPredictor(
            db_manager=mock_db_manager,
            miner_stats_handler=mock_miner_stats_handler,
            preprocessor_path=preprocessor_path,
            team_averages_path=team_averages_path,
            calibrated_model_path=calibrated_model_path,
            predictions_handler=predictions_handler,
        )
        nfl_predictor.preprocessor = self.real_preprocessor
        nfl_predictor.calibrated_model = self.real_sklearn

        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "bettensor",
            "miner",
            "models",
            "nfl_wager_model.pt",
        )
        print(f"Attempting to load model from: {model_path}")
        print(f"Directory exists: {os.path.isdir(model_path)}")

        if os.path.isdir(model_path):
            nfl_predictor.model = KellyFractionNet.from_pretrained(model_path)
            nfl_predictor.model.eval()
        else:
            raise FileNotFoundError(
                f"The model directory does not exist at {model_path}"
            )

        predictions_handler.models["nfl"] = nfl_predictor

        current_time = datetime.now(timezone.utc).isoformat()
        games = {
            "game1": TeamGame(
                game_id="game1",
                team_a="New England Patriots",
                team_b="Buffalo Bills",
                team_a_odds=3.8,
                team_b_odds=3.1,
                tie_odds=1.0,
                sport="nfl",
                league="NFL",
                external_id="ext1",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
            "game2": TeamGame(
                game_id="game2",
                team_a="Green Bay Packers",
                team_b="Chicago Bears",
                team_a_odds=1.9,
                team_b_odds=2.0,
                tie_odds=3.0,
                sport="nfl",
                league="NFL",
                external_id="ext2",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
            "game3": TeamGame(
                game_id="game3",
                team_a="Dallas Cowboys",
                team_b="Philadelphia Eagles",
                team_a_odds=2.2,
                team_b_odds=1.7,
                tie_odds=3.5,
                sport="nfl",
                league="NFL",
                external_id="ext3",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
            "game4": TeamGame(
                game_id="game4",
                team_a="Kansas City Chiefs",
                team_b="Las Vegas Raiders",
                team_a_odds=2.5,
                team_b_odds=1.3,
                tie_odds=4.0,
                sport="nfl",
                league="NFL",
                external_id="ext4",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
            "game5": TeamGame(
                game_id="game5",
                team_a="San Francisco 49ers",
                team_b="Seattle Seahawks",
                team_a_odds=1.6,
                team_b_odds=2.5,
                tie_odds=3.8,
                sport="nfl",
                league="NFL",
                external_id="ext5",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
        }

        predictions = predictions_handler.process_model_predictions(games, "nfl")
        print("Processed predictions:")
        for game_id, prediction in predictions.items():
            print(f"Game {game_id}:")
            print(f"  Predicted Outcome: {prediction.predicted_outcome}")
            print(f"  Wager: {prediction.wager}")
            print(f"  Team A: {prediction.team_a}")
            print(f"  Team B: {prediction.team_b}")
            print(f"  Team A Odds: {prediction.team_a_odds}")
            print(f"  Team B Odds: {prediction.team_b_odds}")
            print()

        self.assertEqual(len(predictions), 5)
        for game_id, prediction in predictions.items():
            self.assertIsInstance(prediction, TeamGamePrediction)
            self.assertIn(
                prediction.predicted_outcome, [prediction.team_a, prediction.team_b]
            )
            self.assertGreaterEqual(prediction.wager, 0)
            self.assertLess(prediction.wager, 1000)

        print("Raw predictions from NFLPredictor:")
        raw_predictions = nfl_predictor.predict_games(
            home_teams=[game.team_a for game in games.values()],
            away_teams=[game.team_b for game in games.values()],
            odds=[
                [game.team_a_odds, game.tie_odds, game.team_b_odds]
                for game in games.values()
            ],
        )
        for pred in raw_predictions:
            print(pred)

    def test_get_best_match_nfl(self):
        self.predictions_handler.models["nfl"] = MagicMock()
        self.predictions_handler.models["nfl"].fuzzy_match_percentage = 90
        encoded_teams = {
            "New England Patriots",
            "Buffalo Bills",
            "Green Bay Packers",
            "Chicago Bears",
        }

        match = self.predictions_handler.get_best_match(
            "New England Patriots", encoded_teams, "nfl"
        )
        self.assertEqual(match, "New England Patriots")

        match = self.predictions_handler.get_best_match(
            "New England", encoded_teams, "nfl"
        )
        self.assertEqual(match, "New England Patriots")

        match = self.predictions_handler.get_best_match(
            "Los Angeles Lakers", encoded_teams, "nfl"
        )
        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()