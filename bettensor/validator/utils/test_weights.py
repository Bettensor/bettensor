import unittest
import sqlite3
import torch
import tempfile
import os
from datetime import datetime, timedelta, timezone
from bettensor.utils.weights_functions import WeightSetter
import random

class MockMetagraph:
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys
        self.S = torch.ones(len(hotkeys))
        self.uids = list(range(len(hotkeys)))

class TestWeightSetter(unittest.TestCase):
    def setUp(self):
        # Creates a temp database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Creates necessary tables
        self.create_tables()
        
        # Inserts mock data
        self.insert_mock_data()
        
        # Creates mock objects
        self.hotkeys = [f'hotkey{i}' for i in range(1, 51)]
        self.mock_metagraph = MockMetagraph(self.hotkeys)
        self.mock_wallet = type('MockWallet', (), {'hotkey': type('MockHotkey', (), {'ss58_address': 'hotkey1'})()})()
        self.mock_subtensor = None
        self.mock_neuron_config = type('MockNeuronConfig', (), {'netuid': 1})()
        
        # Creates WeightSetter instance
        self.weight_setter = WeightSetter(
            self.mock_metagraph,
            self.mock_wallet,
            self.mock_subtensor,
            self.mock_neuron_config,
            None,
            None,
            self.db_path
        )
    
    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_miner_stats (
                date DATE,
                minerId TEXT,
                total_predictions INT,
                correct_predictions INT,
                total_wager REAL,
                total_earnings REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                predictionDate TEXT,
                minerId TEXT,
                predictedOutcome TEXT,
                outcome TEXT,
                wager REAL,
                teamAodds REAL,
                teamBodds REAL,
                tieOdds REAL
            )
        ''')
        self.conn.commit()
    
    def insert_mock_data(self):
        # Generates 50 unique hotkeys
        hotkeys = [f'hotkey{i}' for i in range(1, 51)]
        
        # Inserts mock daily stats
        today = datetime.now(timezone.utc).date()
        for i in range(30):  # Data for the last 30 days
            date = (today - timedelta(days=i)).isoformat()
            for hotkey in hotkeys:
                # Generate random data for each miner
                total_predictions = random.randint(50, 500)
                correct_predictions = random.randint(0, total_predictions)
                total_wager = random.uniform(100, 5000)
                total_earnings = random.uniform(0, total_wager * 2)  # Can win up to 2x the wager
                
                self.cursor.execute('''
                    INSERT INTO daily_miner_stats (date, minerId, total_predictions, correct_predictions, total_wager, total_earnings)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (date, hotkey, total_predictions, correct_predictions, total_wager, total_earnings))
        
        # Insert mock predictions for today
        for hotkey in hotkeys:
            predicted_outcome = random.choice(['0', '1', '2'])
            outcome = random.choice(['0', '1', '2'])
            wager = random.uniform(10, 100)
            team_a_odds = random.uniform(1.1, 3.0)
            team_b_odds = random.uniform(1.1, 3.0)
            tie_odds = random.uniform(2.0, 5.0)
            
            self.cursor.execute('''
                INSERT INTO predictions (predictionDate, minerId, predictedOutcome, outcome, wager, teamAodds, teamBodds, tieOdds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (today.isoformat(), hotkey, predicted_outcome, outcome, wager, team_a_odds, team_b_odds, tie_odds))
        
        self.conn.commit()
    
    def test_weight_calculation(self):
        earnings = self.weight_setter.calculate_miner_scores(self.db_path)
        weights = torch.nn.functional.normalize(earnings, p=1.0, dim=0)
        
        # Check if weights sum to 1
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        
        # Check if all weights are non-negative
        self.assertTrue(torch.all(weights >= 0))
        
        # Check if at least one weight is non-zero
        self.assertTrue(torch.any(weights > 0))
        
        # Print statistics
        print(f"Max weight: {weights.max().item()}")
        print(f"Min non-zero weight: {weights[weights > 0].min().item()}")
        print(f"Mean weight: {weights.mean().item()}")
        print(f"Median weight: {weights.median().item()}")
        
        # Print all weights
        print("\nAll weights:")
        sorted_weights, sorted_indices = torch.sort(weights, descending=True)
        for i, (value, index) in enumerate(zip(sorted_weights, sorted_indices)):
            print(f"{i+1}. Hotkey: {self.hotkeys[index]}, Weight: {value.item():.8f}")
        
        # Print number of zero weights
        zero_weights = (weights == 0).sum().item()
        print(f"\nNumber of zero weights: {zero_weights}")
        
        # Print number of non-zero weights
        non_zero_weights = (weights > 0).sum().item()
        print(f"Number of non-zero weights: {non_zero_weights}")

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)  # Delete the temp database file

if __name__ == '__main__':
    unittest.main()