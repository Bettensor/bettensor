import unittest
import tempfile
import os
import torch as t
import bittensor as bt
from datetime import datetime, timedelta, timezone
from bettensor.validator.utils.scoring.entropy_system import EntropySystem
from bettensor.validator.utils.scoring.scoring import ScoringSystem
from bettensor.validator.utils.scoring.scoring_data import ScoringData
import random
import logging

class TestScoringSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Debug: Starting setUpClass method")
        logging.debug("Starting setUpClass method")

        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_scoring.db")
        print(f"Debug: Database path: {cls.db_path}")
        logging.debug(f"Database path: {cls.db_path}")

        cls.num_miners = 256
        cls.max_days = 45
        cls.scores_per_day = 24

        cls.initial_date = datetime(
            2023, 5, 1, tzinfo=timezone.utc
        )  # Define initial_date
        cls.scoring_system = ScoringSystem(
            cls.db_path,
            cls.num_miners,
            cls.max_days,
            reference_date=cls.initial_date,  # Pass reference_date
        )



        # Verify that the game_data table exists
        table_check = cls.scoring_system.scoring_data.db_manager.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_data'", ()
        )
        if not table_check:
            raise Exception("game_data table does not exist in the database")

        # Add this check
        if not os.path.exists(cls.db_path):
            print(f"Warning: Database file does not exist at {cls.db_path}")
        
        # Check if the table exists
        table_check = cls.scoring_system.scoring_data.db_manager.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_data'", ()
        )
        if not table_check:
            print("Warning: game_data table does not exist in the database")

        print("Debug: Calling _add_test_data")
        logging.debug("Calling _add_test_data")
        cls._add_test_data()

        # Fetch prediction count using DatabaseManager
        prediction_count = cls.scoring_system.scoring_data.db_manager.fetch_one(
            "SELECT COUNT(*) FROM predictions", ()
        )
        print(f"Debug: Total predictions in database: {prediction_count}")

        logging.basicConfig(level=logging.DEBUG)
        cls.logger = logging.getLogger(__name__)

        # Run a full 16-day simulation
        cls.run_full_simulation()

        print("Debug: Finished setUpClass method")
        logging.debug("Finished setUpClass method")

    @classmethod
    def tearDownClass(cls):
        for root, dirs, files in os.walk(cls.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(cls.temp_dir)

    @classmethod
    def _add_test_data(cls):
        print("Debug: Starting _add_test_data method")
        logging.debug("Starting _add_test_data method")

        test_dates = [
            (datetime(2023, 5, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
            for i in range(16)  # 16 days of data
        ]
        print(f"Debug: Test dates: {test_dates}")
        logging.debug(f"Test dates: {test_dates}")

        games_per_day = 10
        game_id_counter = 0
        game_data_batch = []
        prediction_data_batch = []

        for test_date in test_dates:
            for _ in range(games_per_day):
                external_id = f"EXT_{game_id_counter}"
                create_date = test_date
                last_update_date = test_date
                
                can_tie = random.choice([True, False])
                if can_tie:
                    outcome = random.randint(0, 2)  # 0 - Team A, 1 - Team B, 2 - Tie
                    tie_odds = random.uniform(2.0, 4.0)
                else:
                    outcome = random.randint(0, 1)  # 0 - Team A, 1 - Team B
                    tie_odds = 0.0  # No tie possible

                team_a_odds = random.uniform(1.5, 3.5)
                team_b_odds = random.uniform(1.5, 3.5)
                
                game_data = (
                    game_id_counter,
                    external_id,
                    "Team A",
                    "Team B",
                    team_a_odds,
                    team_b_odds,
                    tie_odds,
                    can_tie,
                    test_date,
                    create_date,
                    last_update_date,
                    "TestSport",
                    outcome,
                    1,
                )
                game_data_batch.append(game_data)

                for miner_uid in range(cls.num_miners):
                    predicted_outcome = random.randint(0, 2) if can_tie else random.randint(0, 1)
                    if predicted_outcome == 0:
                        odds = team_a_odds
                    elif predicted_outcome == 1:
                        odds = team_b_odds
                    else:
                        odds = tie_odds

                    wager = 1000 / games_per_day
                    prediction_data = (
                        f"{test_date}_{miner_uid}_{external_id}",
                        external_id,
                        miner_uid,
                        test_date,
                        predicted_outcome,
                        random.uniform(1.5, 4.0),
                        wager,
                    )
                    prediction_data_batch.append(prediction_data)

                game_id_counter += 1

        # Batch insert game data
        cls.scoring_system.scoring_data.db_manager.execute_query(
            """
            INSERT INTO game_data (
                id, external_id, team_a, team_b, team_a_odds, team_b_odds, tie_odds,
                can_tie, event_start_date, create_date, last_update_date, sport, outcome, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            game_data_batch,
            batch=True
        )

        # Batch insert prediction data
        cls.scoring_system.scoring_data.db_manager.execute_query(
            """
            INSERT INTO predictions (
                prediction_id, game_id, miner_uid, prediction_date,
                predicted_outcome, predicted_odds, wager
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            prediction_data_batch,
            batch=True
        )

        # Add these debug statements at the end of the method
        print(f"Debug: Total games attempted to insert: {game_id_counter}")
        total_games = cls.scoring_system.scoring_data.db_manager.fetch_one(
            "SELECT COUNT(*) FROM game_data", ()
        )[0]
        total_predictions = cls.scoring_system.scoring_data.db_manager.fetch_one(
            "SELECT COUNT(*) FROM predictions", ()
        )[0]
        print(f"Debug: Total games in database: {total_games}")
        print(f"Debug: Total predictions in database: {total_predictions}")

        # Debug: Check available dates
        available_dates = cls.scoring_system.scoring_data.db_manager.fetch_all(
            "SELECT DISTINCT event_start_date FROM game_data ORDER BY event_start_date",
            ()  # Empty tuple for params
        )
        print(f"Debug: Available dates: {available_dates}")

        print(f"Debug: Added test data for dates: {test_dates}")
        print(f"Debug: Total games added: {game_id_counter}")
        print(f"Debug: Total predictions added: {game_id_counter * cls.num_miners}")
        print(
            f"Debug: Each miner wagered approximately {1000 * len(test_dates)} units in total"
        )
        bt.logging.trace("Test data added successfully.")

        # Add this debug statement at the end of the method
        print(f"Debug: Database path: {cls.db_path}")

    @classmethod
    def run_full_simulation(cls):
        cls.tier_history = []
        cls.score_history = []
        cls.wager_history = []
        cls.weight_history = []

        # Fetch all games at once
        all_games = cls.scoring_system.scoring_data.db_manager.fetch_all(
            "SELECT * FROM game_data ORDER BY event_start_date",
            ()  # Empty tuple for params
        )
        games_by_date = {}
        for game in all_games:
            date = game[8]  # Assuming event_start_date is at index 8
            if date not in games_by_date:
                games_by_date[date] = []
            games_by_date[date].append(game)

        for i in range(16):
            current_date = cls.initial_date + timedelta(days=i)
            bt.logging.info(f"Running simulation for day {i}")
            
            # Debug: Check for games on the current date
            games_on_date = games_by_date.get(current_date.isoformat(), [])
            print(f"Debug: Games on {current_date.isoformat()}: {len(games_on_date)}")
            
            weights = cls.scoring_system.scoring_run(current_date)
            bt.logging.info(f"Weights: {weights}")

            current_tiers = cls.scoring_system._get_tensor_for_day(
                cls.scoring_system.tiers,
                cls.scoring_system.current_day,
            ).clone()
            tier_distribution = [
                int((current_tiers == tier).sum().item())
                for tier in range(1, len(cls.scoring_system.tier_configs) + 1)
            ]
            cls.tier_history.append(tier_distribution)

            current_scores = cls.scoring_system._get_tensor_for_day(
                cls.scoring_system.composite_scores,
                cls.scoring_system.current_day,
            )
            avg_score = current_scores.mean().item()
            cls.score_history.append(avg_score)

            current_wagers = (
                cls.scoring_system._get_tensor_for_day(
                    cls.scoring_system.amount_wagered,
                    cls.scoring_system.current_day,
                )
                .sum()
                .item()
            )
            cls.wager_history.append(current_wagers)

            cls.weight_history.append(weights)

            cls.logger.debug(f"Day {i}: Tier distribution: {tier_distribution}")
            cls.logger.debug(f"Day {i}: Average score: {avg_score:.4f}")
            cls.logger.debug(f"Day {i}: Total wager: {current_wagers:.2f}")
            cls.logger.debug(
                f"Day {i}: Non-zero scores: {(current_scores != 0).sum().item()}"
            )
            cls.logger.debug(
                f"Day {i}: Min score: {current_scores.min().item():.4f}, Max score: {current_scores.max().item():.4f}"
            )
            bt.logging.info(f"Completed simulation for day {i}")

    def test_preprocess_for_scoring(self):
        # Use ISO format for the date
        date = self.initial_date.isoformat()
        print(f"Debug: Testing for date: {date}")

        (
            predictions,
            closing_line_odds,
            results,
        ) = self.scoring_system.scoring_data.preprocess_for_scoring(date)

        print(f"Debug: Number of predictions: {len(predictions)}")
        print(f"Debug: Shape of closing_line_odds: {closing_line_odds.shape}")
        print(f"Debug: Shape of results: {results.shape}")

        # Check if there are any games for this date
        games_on_date = self.scoring_system.scoring_data.db_manager.fetch_all(
            "SELECT COUNT(*) FROM game_data WHERE event_start_date = ?",
            (date,)
        )
        print(f"Debug: Number of games on {date}: {games_on_date[0][0]}")

        # Check if there are any predictions for this date
        predictions_on_date = self.scoring_system.scoring_data.db_manager.fetch_all(
            "SELECT COUNT(*) FROM predictions WHERE prediction_date = ?",
            (date,)
        )
        print(f"Debug: Number of predictions on {date}: {predictions_on_date[0][0]}")

        self.assertGreater(len(predictions), 0, f"No predictions found for date {date}")
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(predictions[0], t.Tensor)
        self.assertEqual(
            predictions[0].shape[1], 4
        )  # [game_id, predicted_outcome, predicted_odds, wager]

        self.assertIsInstance(closing_line_odds, t.Tensor)
        self.assertEqual(
            closing_line_odds.shape[1], 4
        )  # [game_id, team_a_odds, team_b_odds, tie_odds]

        self.assertIsInstance(results, t.Tensor)
        self.assertEqual(results.shape[0], closing_line_odds.shape[0])

    def test_entropy_system_update_ebdr_scores(self):
        date = self.initial_date.isoformat()
        predictions, closing_line_odds, results = self.scoring_system.scoring_data.preprocess_for_scoring(date)

        entropy_system = self.scoring_system.entropy_system
        ebdr_scores = entropy_system.update_ebdr_scores(predictions, closing_line_odds, results)

        self.assertIsInstance(ebdr_scores, t.Tensor)
        self.assertEqual(ebdr_scores.shape, (self.num_miners, self.max_days))
        self.assertTrue(t.any(ebdr_scores != 0), "All EBDR scores are zero.")
        print(f"Number of non-zero EBDR scores: {(ebdr_scores != 0).sum().item()}")
        print(f"Game entropies: {entropy_system.game_outcome_entropies}")

        # Verify sum of entropy scores for a miner
        miner_entropy_sum = ebdr_scores[:, self.scoring_system.current_day]
        self.assertTrue(t.all(miner_entropy_sum >= 0), "Some miner entropy sums are negative.")
        print(f"Miner entropy sums: {miner_entropy_sum}")

        # Verify that each prediction is scored upon submission
        for i, miner_predictions in enumerate(predictions):
            if miner_predictions.numel() > 0:
                for pred in miner_predictions:
                    game_id, outcome, odds, wager = pred
                    game_id = int(game_id.item())
                    self.assertLess(game_id, closing_line_odds.shape[0], f"Game ID {game_id} out of range.")
                    self.assertIn(game_id, entropy_system.game_outcome_entropies, f"Game {game_id} entropy not found.")
                    game_entropy = entropy_system.game_outcome_entropies[game_id]
                    self.assertIsInstance(game_entropy, dict, f"Game {game_id} entropy is not a dictionary.")
                    self.assertTrue(any(v > 0 for v in game_entropy.values()), f"Game {game_id} has no positive entropy values.")

        # Print additional debug information
        print(f"EBDR scores shape: {ebdr_scores.shape}")
        print(f"EBDR scores non-zero count: {(ebdr_scores != 0).sum().item()}")
        print(f"EBDR scores statistics: Min: {ebdr_scores.min().item():.8f}, Max: {ebdr_scores.max().item():.8f}, Mean: {ebdr_scores.mean().item():.8f}")

    def test_update_new_day(self):
        # Check tier changes over the first 5 days
        initial_tiers = self.scoring_system._get_tensor_for_day(
            self.scoring_system.tiers, 0
        )

        tiers_changed = False
        for day in range(1, 5):  # Check days 1 to 4
            current_tiers = self.scoring_system._get_tensor_for_day(
                self.scoring_system.tiers, day
            )
            if not t.all(initial_tiers == current_tiers):
                tiers_changed = True
                break

        self.assertTrue(tiers_changed, "Tiers did not update within the first 5 days")

        if tiers_changed:
            print(f"Tiers changed on day {day}")
            print("Initial tier distribution:", self.tier_history[0])
            print(f"Day {day} tier distribution:", self.tier_history[day])
        else:
            print("Tier distributions for the first 5 days:")
            for i in range(5):
                print(f"Day {i}:", self.tier_history[i])

    def test_calculate_weights(self):
        # This test can use the data from the last day of the simulation
        weights = self.weight_history[-1]

        self.assertEqual(weights.shape, t.Size([self.num_miners]))
        self.assertTrue(t.all(weights >= 0))
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)

    def test_tier_progression(self):
        initial_distribution = self.tier_history[0]
        final_distribution = self.tier_history[-1]

        self.assertNotEqual(
            initial_distribution,
            final_distribution,
            "Tier distribution did not change over time",
        )

        # Check that we have miners in at least tier 3 by the end
        self.assertTrue(
            any(count > 0 for count in final_distribution[2:]),
            "No miners reached tier 3 or above after 16 days",
        )

        print("\nTier progression over 16 days:")
        for day, (distribution, avg_score, total_wager) in enumerate(
            zip(self.tier_history, self.score_history, self.wager_history)
        ):
            print(
                f"Day {day}: {distribution}, Avg Score: {avg_score:.4f}, Total Wager: {total_wager:.2f}"
            )

    def test_score_stability(self):
        # Check that scores are relatively stable over time
        score_diff = self.score_history[-1] - self.score_history[0]
        self.assertLess(abs(score_diff), 0.1, "Scores are not stable over time")

    def test_wager_increase(self):
        # Check that wagers are increasing over time
        self.assertGreater(
            sum(self.wager_history[7:]),
            sum(self.wager_history[:7]),
            "Wagers are not increasing over time",
        )

    def test_entropy_system_integration(self):
        final_entropy_scores = self.scoring_system.entropy_scores[:, self.scoring_system.current_day]

        print(f"Entropy scores shape: {final_entropy_scores.shape}")
        print(f"Entropy scores non-zero count: {(final_entropy_scores != 0).sum().item()}")
        print(f"Entropy scores statistics: Min: {final_entropy_scores.min().item():.8f}, Max: {final_entropy_scores.max().item():.8f}, Mean: {final_entropy_scores.mean().item():.8f}")
        print(f"Number of game entropies: {len(self.scoring_system.entropy_system.game_outcome_entropies)}")
        print(f"Game entropies: {self.scoring_system.entropy_system.game_outcome_entropies}")

        self.assertTrue(
            t.any(final_entropy_scores >= 0),
            f"No non-negative entropy scores. Min: {final_entropy_scores.min().item():.8f}, Max: {final_entropy_scores.max().item():.8f}, Mean: {final_entropy_scores.mean().item():.8f}",
        )

    def test_wraparound(self):
        # This test can use the data from the full simulation
        expected_day = 15 % self.scoring_system.max_days  # 15 is the last day of our 16-day simulation
        self.assertEqual(self.scoring_system.current_day, expected_day)

    def test_window_calculation(self):
        # This test might need to be adjusted to use the data from the full simulation
        window = 30
        start_day = self.scoring_system.current_day
        window_scores = self.scoring_system._get_window_scores(
            self.scoring_system.clv_scores, start_day, window
        )

        # Adjust the expected_scores calculation based on the actual data from the simulation
        expected_days = [
            (start_day - offset) % self.scoring_system.max_days
            for offset in range(window)
        ]
        expected_scores = self.scoring_system.clv_scores[:, expected_days]

        self.assertEqual(
            window_scores.shape[1],
            window,
            f"Expected window_scores to have {window} elements, got {window_scores.shape[1]}",
        )
        self.assertTrue(
            t.allclose(window_scores, expected_scores, atol=1e-4),
            f"Window scores do not match expected scores.\n"
            f"Window scores: {window_scores}\n"
            f"Expected scores: {expected_scores}\n"
            f"Difference: {(window_scores - expected_scores).abs().max()}",
        )

        # Print shapes for debugging
        print(f"window_scores shape: {window_scores.shape}")
        print(f"expected_scores shape: {expected_scores.shape}")

    def test_database_connection(self):
        print("Debug: Starting test_database_connection")
        try:
            result = self.scoring_system.scoring_data.db_manager.fetch_one(
                "SELECT COUNT(*) FROM game_data", ()
            )
            print(f"Debug: Test database query result: {result}")
            self.assertIsNotNone(result)
        except Exception as e:
            print(f"Error in test_database_connection: {e}")
            raise

if __name__ == "__main__":
    unittest.main()
