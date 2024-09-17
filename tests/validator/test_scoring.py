import unittest
import tempfile
import os
import torch as t
from datetime import datetime, timedelta, timezone
from bettensor.validator.utils.scoring.entropy_system import EntropySystem
from bettensor.validator.utils.scoring.scoring import ScoringSystem
from bettensor.validator.utils.scoring.scoring_data import ScoringData

class TestScoringSystem(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_scoring.db')
        
        self.num_miners = 256
        self.max_days = 45
        
        # Initialize ScoringSystem with the test database
        self.scoring_system = ScoringSystem(self.db_path, self.num_miners, self.max_days)
        
        # Add test data
        self._add_test_data()
        
        # Verify data insertion
        conn = self.scoring_system.scoring_data.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        prediction_count = cursor.fetchone()[0]
        print(f"Debug: Total predictions in database: {prediction_count}")
        conn.close()
        
        # Ensure the scoring system is initialized with the test data
        self.scoring_system.scoring_data.preprocess_for_scoring(datetime.now().strftime('%Y-%m-%d'))

    def tearDown(self):
        # Remove the temporary directory and its contents
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

    def _add_test_data(self):
        conn = self.scoring_system.scoring_data.connect_db()
        cursor = conn.cursor()
        
        test_dates = [
            datetime(2023, 5, 1).strftime('%Y-%m-%d'),
            datetime(2023, 5, 2).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ]
        
        game_id_counter = 0
        for test_date in test_dates:
            # Add test games
            for _ in range(10):
                external_id = f"EXT_{game_id_counter}"
                cursor.execute("""
                    INSERT OR IGNORE INTO game_data (id, external_id, team_a, team_b, team_a_odds, team_b_odds, tie_odds, event_start_date, sport, outcome, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id_counter, external_id, "Team A", "Team B", 2.0, 2.0, 3.0, test_date, "TestSport", 0, 1))
                game_id_counter += 1

            # Add predictions for each miner
            for miner_uid in range(self.num_miners):
                for game_id in range(game_id_counter - 10, game_id_counter):
                    external_id = f"EXT_{game_id}"
                    cursor.execute("""
                        INSERT OR IGNORE INTO predictions (prediction_id, game_id, miner_uid, prediction_date, predicted_outcome, predicted_odds, wager)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (f"{test_date}_{miner_uid}_{external_id}", external_id, miner_uid, test_date, 0, 2.0, 1.0))

        conn.commit()
        conn.close()

        print(f"Debug: Added test data for dates: {test_dates}")

    def test_update_clv(self):
        predictions = [t.rand(10, 3) for _ in range(self.num_miners)]
        closing_line_odds = t.rand(10, 2)
        
        clv_scores = self.scoring_system._update_clv(predictions, closing_line_odds)
        
        self.assertEqual(clv_scores.shape, t.Size([self.num_miners, self.max_days]))
        self.assertTrue(t.all(clv_scores[:, -1] >= -100) and t.all(clv_scores[:, -1] <= 100))

    def test_update_roi(self):
        predictions = [t.rand(10, 3) for _ in range(self.num_miners)]
        results = t.randint(0, 2, (10,))
        
        roi_scores = self.scoring_system._update_roi(predictions, results)
        
        self.assertEqual(roi_scores.shape, t.Size([self.num_miners, self.max_days]))
        self.assertTrue(t.all(roi_scores[:, -1] >= -1) and t.all(roi_scores[:, -1] <= 1))

    def test_update_sortino(self):
        self.scoring_system.roi_scores = t.randn(self.num_miners, self.max_days)
        
        sortino_scores = self.scoring_system._update_sortino()
        
        self.assertEqual(sortino_scores.shape, t.Size([self.num_miners, self.max_days]))
        self.assertTrue(t.all(sortino_scores.isfinite()))

    def test_calculate_amount_wagered(self):
        predictions = [t.rand(t.randint(1, 20, (1,)).item(), 3) for _ in range(self.num_miners)]
        
        amount_wagered = self.scoring_system._calculate_amount_wagered(predictions)
        
        self.assertEqual(amount_wagered.shape, t.Size([self.num_miners]))
        self.assertTrue(t.all(amount_wagered >= 0))

    def test_update_composite_scores(self):
        self.scoring_system.clv_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.roi_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.sortino_scores = t.rand(self.num_miners, self.max_days)  # Update to 2D tensor
        self.scoring_system.entropy_scores = t.rand(self.num_miners, self.max_days)
        
        composite_scores = self.scoring_system._update_composite_scores()
        
        self.assertEqual(composite_scores.shape, t.Size([self.num_miners, self.max_days]))
        self.assertTrue(t.all(composite_scores >= 0))

    def test_calculate_composite_scores(self):
        self.scoring_system.clv_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.roi_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.sortino_scores = t.rand(self.num_miners, self.max_days)  # Update to 2D tensor
        self.scoring_system.entropy_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.tiers = t.randint(1, 6, (self.num_miners,))
        
        composite_scores = self.scoring_system.calculate_composite_scores()
        
        self.assertEqual(composite_scores.shape, t.Size([self.num_miners]))
        self.assertTrue(t.all(composite_scores >= 0))

    def test_reset_miner(self):
        miner_uid = t.randint(0, self.num_miners - 1, (1,)).item()
        self.scoring_system.reset_miner(miner_uid)
        
        self.assertEqual(self.scoring_system.clv_scores[miner_uid].sum().item(), 0)
        self.assertEqual(self.scoring_system.sortino_scores[miner_uid].sum().item(), 0)
        self.assertEqual(self.scoring_system.roi_scores[miner_uid].sum().item(), 0)
        self.assertEqual(self.scoring_system.amount_wagered[miner_uid].sum().item(), 0)
        self.assertEqual(self.scoring_system.composite_scores[miner_uid].sum().item(), 0)
        self.assertEqual(self.scoring_system.entropy_scores[miner_uid].sum().item(), 0)
        self.assertEqual(self.scoring_system.tiers[miner_uid].item(), 1)
        self.assertEqual(self.scoring_system.tier_history[miner_uid].sum().item(), self.max_days)

    def test_calculate_composite_score(self):
        miner_uid = t.randint(0, self.num_miners - 1, (1,)).item()
        window = t.randint(1, self.max_days, (1,)).item()
        self.scoring_system.clv_scores[miner_uid, -window:] = t.rand(window)
        self.scoring_system.roi_scores[miner_uid, -window:] = t.rand(window)
        self.scoring_system.sortino_scores[miner_uid, -window:] = t.rand(window)  # Update to 2D tensor
        self.scoring_system.entropy_scores[miner_uid, -window:] = t.rand(window)
        
        score = self.scoring_system.calculate_composite_score(miner_uid, window)
        
        self.assertTrue(0 <= score <= 4)  # Assuming each component is between 0 and 1

    def test_manage_tiers(self):
        # Initialize all miners in tier 1
        self.scoring_system.tiers = t.ones(self.num_miners, dtype=t.int)
        initial_distribution = self.scoring_system.tiers.clone()
        
        # Simulate some activity
        self.scoring_system.amount_wagered = t.rand(self.num_miners, self.max_days) * 1000
        self.scoring_system.clv_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.roi_scores = t.rand(self.num_miners, self.max_days) * 0.2 - 0.1  # ROI between -10% and 10%
        self.scoring_system.sortino_scores = t.rand(self.num_miners, self.max_days)

        self.scoring_system.manage_tiers()
        
        final_distribution = self.scoring_system.tiers.clone()
        
        self.assertFalse(t.all(initial_distribution == final_distribution), 
                         "Tier distribution did not change after manage_tiers")
        
        for tier, config in enumerate(self.scoring_system.tier_configs, 1):
            tier_count = (self.scoring_system.tiers == tier).sum().item()
            self.assertLessEqual(tier_count, config['capacity'], 
                                 f"Tier {tier} exceeds capacity ({tier_count} > {config['capacity']})")

    def test_manage_tiers_debug_output(self):
        self.scoring_system.tiers = t.ones(self.num_miners, dtype=t.int)
        self.scoring_system.amount_wagered = t.rand(self.num_miners, self.max_days) * 1000
        self.scoring_system.clv_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.roi_scores = t.rand(self.num_miners, self.max_days) * 0.2 - 0.1
        self.scoring_system.sortino_scores = t.rand(self.num_miners, self.max_days)
        
        # Ensure some miners have high scores to trigger promotions
        self.scoring_system.composite_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.composite_scores[:10, :] = 0.9  # Set high scores for first 10 miners
        
        # Ensure some miners have low scores to trigger demotions
        self.scoring_system.composite_scores[-10:, :] = 0.1  # Set low scores for last 10 miners
        
        # Set some miners to higher tiers to allow for demotions
        self.scoring_system.tiers[-10:] = 2

        with self.assertLogs(self.scoring_system.logger, level='INFO') as cm:
            self.scoring_system.manage_tiers(debug=True)
        
        self.assertTrue(any("promoted to tier" in log for log in cm.output), "No promotions occurred")
        self.assertTrue(any("demoted from tier" in log for log in cm.output), "No demotions occurred")

    def test_calculate_weights(self):
        self.scoring_system.clv_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.roi_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.sortino_scores = t.rand(self.num_miners, self.max_days)  # Update to 2D tensor
        self.scoring_system.entropy_scores = t.rand(self.num_miners, self.max_days)
        self.scoring_system.tiers = t.randint(1, 6, (self.num_miners,))
        
        weights = self.scoring_system.calculate_weights()
        
        self.assertEqual(weights.shape, t.Size([self.num_miners]))
        self.assertTrue(t.all(weights >= 0))
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)

    def test_scoring_run(self):
        current_date = datetime.now().strftime('%Y-%m-%d')
        weights = self.scoring_system.scoring_run(current_date)
        if weights is None:
            print(f"Debug: No weights returned for date {current_date}")
            # Fetch some debug information
            conn = self.scoring_system.scoring_data.connect_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE DATE(prediction_date) = DATE(?)", (current_date,))
            prediction_count = cursor.fetchone()[0]
            print(f"Debug: Found {prediction_count} predictions for date {current_date}")
            conn.close()
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape[0], self.num_miners)

    def test_multiple_updates_same_day(self):
        initial_date = datetime(2023, 5, 1, tzinfo=timezone.utc)
        self.scoring_system.scoring_run(initial_date)
        
        # Simulate multiple updates on the same day
        for _ in range(5):
            self.scoring_system.scoring_run(initial_date + timedelta(hours=1))
        
        # Check that the tensor hasn't rolled
        self.assertEqual(self.scoring_system.last_update_date, initial_date.date())

    def test_update_new_day(self):
        initial_date = datetime(2023, 5, 1, tzinfo=timezone.utc)
        self.scoring_system.scoring_run(initial_date)
        
        # Simulate an update on the next day
        next_day = initial_date + timedelta(days=1)
        self.scoring_system.scoring_run(next_day)
        
        # Check that the tensor has rolled and the last update date has changed
        self.assertEqual(self.scoring_system.last_update_date, next_day.date())

    def test_initial_entropy_scores(self):
        num_miners = 3
        num_games = 2  # Increase this to 2 to match the shape of predictions
        predictions = [t.tensor([[2.0, 1.0, 0], [3.0, 0.5, 1]]) for _ in range(num_miners)]
        closing_line_odds = t.tensor([[2.0, 3.0], [2.5, 2.5]])  # Add another set of odds
        results = t.tensor([0, 1])  # Add another result

        entropy_system = EntropySystem(max_capacity=num_miners, max_days=self.max_days)
        entropy_scores = entropy_system.update_ebdr_scores(predictions, closing_line_odds, results)

        self.assertIsNotNone(entropy_scores)
        self.assertEqual(entropy_scores.shape, (num_miners, self.max_days))
        self.assertTrue(t.any(entropy_scores != 0))  # Ensure some scores are non-zero

if __name__ == '__main__':
    unittest.main()