import unittest
import tempfile
import os
import torch as t
from datetime import datetime, timedelta, timezone
from bettensor.validator.utils.scoring.entropy_system import EntropySystem
from bettensor.validator.utils.scoring.scoring import ScoringSystem
from bettensor.validator.utils.scoring.scoring_data import ScoringData
import random
import logging

class TestScoringSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, 'test_scoring.db')
        
        cls.num_miners = 256
        cls.max_days = 45
        cls.scores_per_day = 24
        
        cls.initial_date = datetime(2023, 5, 1, tzinfo=timezone.utc)  # Define initial_date
        cls.scoring_system = ScoringSystem(
            cls.db_path, 
            cls.num_miners, 
            cls.max_days,
            reference_date=cls.initial_date  # Pass reference_date
        )
        
        cls._add_test_data()
        
        conn = cls.scoring_system.scoring_data.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        prediction_count = cursor.fetchone()[0]
        print(f"Debug: Total predictions in database: {prediction_count}")
        conn.close()

        logging.basicConfig(level=logging.DEBUG)
        cls.logger = logging.getLogger(__name__)

        # Run a full 16-day simulation
        cls.run_full_simulation()

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
        conn = cls.scoring_system.scoring_data.connect_db()
        cursor = conn.cursor()
        
        test_dates = [
            (datetime(2023, 5, 1, tzinfo=timezone.utc) + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(16)  # 16 days of data
        ]
        
        games_per_day = 10
        game_id_counter = 0
        for test_date in test_dates:
            for _ in range(games_per_day):
                external_id = f"EXT_{game_id_counter}"
                cursor.execute("""
                    INSERT INTO game_data (id, external_id, team_a, team_b, team_a_odds, team_b_odds, tie_odds, event_start_date, sport, outcome, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id_counter, external_id, "Team A", "Team B", 2.0, 2.0, 3.0, test_date, "TestSport", 0, 1))
                
                for miner_uid in range(cls.num_miners):
                    # Calculate wager amount to reach approximately 1000 per day
                    wager = 1000 / games_per_day
                    cursor.execute("""
                        INSERT INTO predictions (prediction_id, game_id, miner_uid, prediction_date, predicted_outcome, predicted_odds, wager)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (f"{test_date}_{miner_uid}_{external_id}", external_id, miner_uid, test_date, random.randint(0, 2), random.uniform(1.5, 3.0), wager))
                
                game_id_counter += 1

        conn.commit()
        conn.close()

        print(f"Debug: Added test data for dates: {test_dates}")
        print(f"Debug: Total games added: {game_id_counter}")
        print(f"Debug: Total predictions added: {game_id_counter * cls.num_miners}")
        print(f"Debug: Each miner wagered approximately {1000 * len(test_dates)} units in total")

    @classmethod
    def run_full_simulation(cls):
        cls.tier_history = []
        cls.score_history = []
        cls.wager_history = []
        cls.weight_history = []

        for i in range(16):
            current_date = cls.initial_date + timedelta(days=i)
            weights = cls.scoring_system.scoring_run(current_date)
            
            current_tiers = cls.scoring_system._get_tensor_for_day(
                cls.scoring_system.tiers, 
                cls.scoring_system.current_day, 
                cls.scoring_system.current_hour
            ).clone()
            tier_distribution = [
                int((current_tiers == tier).sum().item()) 
                for tier in range(1, len(cls.scoring_system.tier_configs) + 1)
            ]
            cls.tier_history.append(tier_distribution)

            current_scores = cls.scoring_system._get_tensor_for_day(
                cls.scoring_system.composite_scores, 
                cls.scoring_system.current_day, 
                cls.scoring_system.current_hour
            )
            avg_score = current_scores.mean().item()
            cls.score_history.append(avg_score)

            current_wagers = cls.scoring_system._get_tensor_for_day(
                cls.scoring_system.amount_wagered, 
                cls.scoring_system.current_day, 
                cls.scoring_system.current_hour
            ).sum().item()
            cls.wager_history.append(current_wagers)

            cls.weight_history.append(weights)

            cls.logger.debug(f"Day {i}: Tier distribution: {tier_distribution}")
            cls.logger.debug(f"Day {i}: Average score: {avg_score:.4f}")
            cls.logger.debug(f"Day {i}: Total wager: {current_wagers:.2f}")
            cls.logger.debug(f"Day {i}: Non-zero scores: {(current_scores != 0).sum().item()}")
            cls.logger.debug(f"Day {i}: Min score: {current_scores.min().item():.4f}, Max score: {current_scores.max().item():.4f}")

    def test_preprocess_for_scoring(self):
        # This test can use the data from the first day of the simulation
        date = self.initial_date.strftime('%Y-%m-%d')
        predictions, closing_line_odds, results = self.scoring_system.scoring_data.preprocess_for_scoring(date)
        
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
        self.assertIsInstance(predictions[0], t.Tensor)
        self.assertEqual(predictions[0].shape[1], 4)  # [game_id, predicted_outcome, predicted_odds, wager]
        
        self.assertIsInstance(closing_line_odds, t.Tensor)
        self.assertEqual(closing_line_odds.shape[1], 4)  # [game_id, team_a_odds, team_b_odds, tie_odds]
        
        self.assertIsInstance(results, t.Tensor)
        self.assertEqual(results.shape[0], closing_line_odds.shape[0])

    def test_entropy_system_update_ebdr_scores(self):
        # This test can use the data from the first day of the simulation
        date = self.initial_date.strftime('%Y-%m-%d')
        predictions, closing_line_odds, results = self.scoring_system.scoring_data.preprocess_for_scoring(date)
        
        entropy_system = EntropySystem(max_capacity=self.num_miners, max_days=self.max_days)
        ebdr_scores = entropy_system.update_ebdr_scores(predictions, closing_line_odds, results)
        
        self.assertIsInstance(ebdr_scores, t.Tensor)
        self.assertEqual(ebdr_scores.shape, (self.num_miners, self.max_days, 24))
        self.assertTrue(t.any(ebdr_scores != 0), f"All EBDR scores are zero. Min: {ebdr_scores.min().item():.8f}, Max: {ebdr_scores.max().item():.8f}, Mean: {ebdr_scores.mean().item():.8f}")
        print(f"Number of non-zero EBDR scores: {(ebdr_scores != 0).sum().item()}")
        print(f"Game entropies: {entropy_system.game_entropies}")

    def test_scoring_run(self):
        # This test can use the data from the first day of the simulation
        weights = self.weight_history[0]
        
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape[0], self.num_miners)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)

    def test_update_new_day(self):
        # Check tier changes over the first 5 days
        initial_tiers = self.scoring_system._get_tensor_for_day(self.scoring_system.tiers, 0, 0)
        
        tiers_changed = False
        for day in range(1, 5):  # Check days 1 to 4
            current_tiers = self.scoring_system._get_tensor_for_day(self.scoring_system.tiers, day, 0)
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

        self.assertNotEqual(initial_distribution, final_distribution, "Tier distribution did not change over time")
        
        # Check that we have miners in at least tier 3 by the end
        self.assertTrue(any(count > 0 for count in final_distribution[2:]), "No miners reached tier 3 or above after 16 days")

        print("\nTier progression over 16 days:")
        for day, (distribution, avg_score, total_wager) in enumerate(zip(self.tier_history, self.score_history, self.wager_history)):
            print(f"Day {day}: {distribution}, Avg Score: {avg_score:.4f}, Total Wager: {total_wager:.2f}")

    def test_score_stability(self):
        # Check that scores are relatively stable over time
        score_diff = self.score_history[-1] - self.score_history[0]
        self.assertLess(abs(score_diff), 0.1, "Scores are not stable over time")

    def test_wager_increase(self):
        # Check that wagers are increasing over time
        self.assertGreater(
            sum(self.wager_history[7:]), 
            sum(self.wager_history[:7]), 
            "Wagers are not increasing over time"
        )

    def test_weight_distribution(self):
        final_weights = self.weight_history[-1]
        self.assertAlmostEqual(final_weights.sum().item(), 1.0, places=6, msg="Weights do not sum to 1")
        self.assertTrue((final_weights >= 0).all(), "Some weights are negative")
        self.assertTrue((final_weights <= 1).all(), "Some weights are greater than 1")

    def test_entropy_system_integration(self):
        final_entropy_scores = self.scoring_system._get_tensor_for_day(
            self.scoring_system.entropy_scores, 
            self.scoring_system.current_day, 
            self.scoring_system.current_hour
        )
        
        print(f"Entropy scores shape: {final_entropy_scores.shape}")
        print(f"Entropy scores non-zero count: {(final_entropy_scores != 0).sum().item()}")
        print(f"Entropy scores statistics: Min: {final_entropy_scores.min().item():.8f}, Max: {final_entropy_scores.max().item():.8f}, Mean: {final_entropy_scores.mean().item():.8f}")
        print(f"Number of game entropies: {len(self.scoring_system.entropy_system.game_entropies)}")
        print(f"Game entropies: {self.scoring_system.entropy_system.game_entropies}")
        
        self.assertTrue(t.any(final_entropy_scores >= 0), f"No non-negative entropy scores. Min: {final_entropy_scores.min().item():.8f}, Max: {final_entropy_scores.max().item():.8f}, Mean: {final_entropy_scores.mean().item():.8f}")

    def test_wraparound(self):
        # This test can use the data from the full simulation
        expected_day = (15) % self.scoring_system.max_days  # 15 is the last day of our 16-day simulation
        self.assertEqual(self.scoring_system.current_day, expected_day)
        self.assertEqual(self.scoring_system.current_hour, 0)

    def test_window_calculation(self):
        # This test might need to be adjusted to use the data from the full simulation
        window = 30
        start_day = self.scoring_system.current_day
        start_hour = 0
        window_scores = self.scoring_system._get_window_scores(
            self.scoring_system.clv_scores, 
            start_day, 
            start_hour, 
            window
        )
        
        # Adjust the expected_scores calculation based on the actual data from the simulation
        expected_days = [(start_day - offset) % self.scoring_system.max_days for offset in range(window)]
        expected_scores = self.scoring_system.clv_scores[:, expected_days, start_hour]
        
        self.assertEqual(
            window_scores.shape[1], 
            window,
            f"Expected window_scores to have {window} elements, got {window_scores.shape[1]}"
        )
        self.assertTrue(
            t.allclose(window_scores, expected_scores, atol=1e-4),
            f"Window scores do not match expected scores.\n"
            f"Window scores: {window_scores}\n"
            f"Expected scores: {expected_scores}\n"
            f"Difference: {(window_scores - expected_scores).abs().max()}"
        )
    
        # Print shapes for debugging
        print(f"window_scores shape: {window_scores.shape}")
        print(f"expected_scores shape: {expected_scores.shape}")

   
if __name__ == '__main__':
    unittest.main()