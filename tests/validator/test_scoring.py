import unittest
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta, timezone
from bettensor.validator.utils.scoring.scoring import ScoringSystem
from bettensor.validator.utils.scoring.scoring_data import ScoringData
from bettensor.validator.utils.database.database_init import initialize_database
from bettensor.validator.utils.database.database_manager import DatabaseManager
import bittensor as bt


class TestScoringSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)  # For reproducibility
        bt.logging.set_trace(True)
        bt.logging.debug("Setting up TestScoringSystem")

        cls.num_miners = 256
        cls.max_days = 45
        cls.simulation_days = 60  # Extended to 60 days

        # Define simulation start and end dates
        cls.simulation_start_date = datetime.now(timezone.utc)

        bt.logging.debug(f"Simulation Start Date: {cls.simulation_start_date}")

        # Use in-memory database
        cls.db_path = "test_database.db"
        cls.db_manager = DatabaseManager(cls.db_path)

        # Verify that tables are created
        cls.db_manager.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        tables = cls.db_manager.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table';", ()
        )
        bt.logging.debug(f"Tables in database: {tables}")

        # Initialize ScoringData with the DatabaseManager
        cls.scoring_data = ScoringData(
            db_manager=cls.db_manager, num_miners=cls.num_miners
        )

        # Initialize ScoringSystem with the ScoringData
        cls.scoring_system = ScoringSystem(
            db_manager=cls.db_manager, num_miners=cls.num_miners, max_days=cls.max_days
        )

        # Set reference date for scoring
        cls.reference_date = datetime.now(timezone.utc)
        cls.scoring_system.reference_date = cls.reference_date

        # Simulation parameters
        cls.valid_uids, cls.invalid_uids = cls._simulate_games_and_predictions()

        # Log the created games
        game_count = cls.db_manager.fetch_one("SELECT COUNT(*) FROM game_data", ())
        bt.logging.debug(f"Total games created: {game_count}")

        # Log a sample of created games
        sample_games = cls.db_manager.fetch_all(
            "SELECT game_id, external_id, event_start_date, active FROM game_data LIMIT 5",
            (),
        )

        bt.logging.debug(f"Sample games: {sample_games}")

        # After simulation
        bt.logging.debug("Verifying inserted games and predictions")

        games = cls.db_manager.execute_query("SELECT COUNT(*) as count FROM game_data")
        predictions = cls.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM predictions"
        )
        # bt.logging.debug(f"Total games inserted: {games}")
        # bt.logging.debug(f"Total predictions inserted: {predictions}")

    @classmethod
    def tearDownClass(cls):
        cls.db_manager.conn.close()
        if os.path.exists("test_database.db"):
            os.remove("test_database.db")
        bt.logging.debug("Test Database Closed and Unlinked")

    @classmethod
    def _simulate_games_and_predictions(cls):
        bt.logging.debug("Starting game and prediction simulation")
        cls.daily_external_ids = [[] for _ in range(cls.simulation_days)]
        current_date = cls.simulation_start_date
        invalid_uids = set(
            np.random.choice(range(cls.num_miners), size=25, replace=False)
        )
        valid_uids = set(range(cls.num_miners)) - invalid_uids

        # Simulate games for each day
        game_data = []
        for day in range(cls.simulation_days):
            current_date = cls.simulation_start_date + timedelta(days=day)
            num_games = max(1, np.random.randint(1, 6))
            for game_num in range(num_games):
                external_id = day * 1000 + game_num
                cls.daily_external_ids[day].append(external_id)
                can_tie = bool(np.random.choice([True, False]))
                tie_odds = round(np.random.uniform(2.0, 5.5), 2) if can_tie else None
                num_outcomes = 3 if can_tie else 2
                outcome = (
                    np.random.randint(0, num_outcomes)
                )
                team_a_odds = round(np.random.uniform(1.5, 4.5), 2)
                team_b_odds = round(np.random.uniform(1.5, 4.5), 2)

                # Correctly initialize the game with the appropriate number of outcomes
                
                odds = (
                    [team_a_odds, team_b_odds, tie_odds]
                    if can_tie
                    else [team_a_odds, team_b_odds]
                )
                cls.scoring_system.entropy_system.add_new_game(
                    external_id, num_outcomes=num_outcomes, odds=odds
                )

                game_data.append(
                    (
                        f"game_{external_id}",
                        external_id,
                        f"TeamA_{external_id}",
                        f"TeamB_{external_id}",
                        team_a_odds,
                        team_b_odds,
                        tie_odds,
                        can_tie,
                        current_date.isoformat(),
                        current_date.isoformat(),
                        current_date.isoformat(),
                        "SportX",
                        "LeagueY",
                        outcome,
                        0,
                    )
                )

        # Batch insert for games
        try:
            rows_affected = cls.db_manager.executemany(
                """
                INSERT INTO game_data (
                    game_id, external_id, team_a, team_b, team_a_odds, team_b_odds, 
                    tie_odds, can_tie, event_start_date, create_date, 
                    last_update_date, sport, league, outcome, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                game_data,
            )
            bt.logging.debug(f"Inserted {rows_affected} games into the database")
        except Exception as e:
            bt.logging.error(f"Error inserting games: {e}")
            raise

        # Verify game insertion
        game_count = cls.db_manager.fetch_one("SELECT COUNT(*) FROM game_data", ())
        bt.logging.debug(f"Number of games in database after insertion: {game_count}")

        # Simulate predictions for each miner per day
        prediction_data = []
        for miner_uid in valid_uids:
            for day in range(cls.simulation_days):
                current_date = cls.simulation_start_date + timedelta(days=day)
                daily_ids = cls.daily_external_ids[day]
                if not daily_ids:
                    bt.logging.debug(f"No games for the day {day}")
                    continue  # Skip if no games for the day
                num_predictions = np.random.randint(
                    1, 5
                )  # 1 to 4 predictions per day per miner
                total_wager = 0
                for pred_num in range(num_predictions):
                    if total_wager >= 1000:
                        break
                    wager = round(np.random.uniform(50, 500), 2)
                    wager = min(wager, 1000 - total_wager)
                    total_wager += wager
                    external_id = int(np.random.choice(daily_ids))
                    bt.logging.debug(f"Selected game {external_id} for prediction")
                    # Retrieve can_tie for the selected game
                    can_tie_game = cls.verify_game_config(external_id)
                    if can_tie_game is None:
                        bt.logging.error(f"Unable to verify configuration for game {external_id}")
                        continue  # Skip this prediction
                    
                    if can_tie_game:
                        predicted_outcome = np.random.randint(0, 3)  # 0, 1, or 2 (tie)
                    else:
                        predicted_outcome = np.random.randint(0, 2)  # 0 or 1

                    team_a_odds = cls.db_manager.fetch_one(
                        "SELECT team_a_odds FROM game_data WHERE external_id = ?",
                        (external_id,),
                    )["team_a_odds"]
                    team_b_odds = cls.db_manager.fetch_one(
                        "SELECT team_b_odds FROM game_data WHERE external_id = ?",
                        (external_id,),
                    )["team_b_odds"]
                    tie_odds = cls.db_manager.fetch_one(
                        "SELECT tie_odds FROM game_data WHERE external_id = ?",
                        (external_id,),
                    )["tie_odds"]

                    # Add some variance in odds to simulate for clv calculation, ensuring odds never go below 1.01
                    team_a_odds = max(
                        1.01, round(team_a_odds + np.random.uniform(-1.9, 1.9), 2)
                    )
                    team_b_odds = max(
                        1.01, round(team_b_odds + np.random.uniform(-1.9, 1.9), 2)
                    )
                    tie_odds = (
                        max(1.01, round(tie_odds + np.random.uniform(-1.9, 1.9), 2))
                        if tie_odds
                        else 0.0
                    )

                    # Determine predicted_odds based on predicted_outcome
                    if predicted_outcome == 0:
                        predicted_odds = team_a_odds
                    elif predicted_outcome == 1:
                        predicted_odds = team_b_odds
                    elif predicted_outcome == 2:
                        predicted_odds = tie_odds

                    prediction_date = (
                        current_date + timedelta(days=np.random.randint(-2, 0))
                    ).isoformat()

                    prediction_data.append(
                        (
                            f"pred_{miner_uid}_{day}_{pred_num}",
                            external_id,
                            miner_uid,
                            prediction_date,
                            predicted_outcome,
                            predicted_odds,
                            f"TeamA_{external_id}",
                            f"TeamB_{external_id}",
                            wager,
                            team_a_odds,
                            team_b_odds,
                            tie_odds,
                            False,
                            None,
                            None,
                            0,
                        )
                    )

                    cls.scoring_system.entropy_system.add_prediction(
                        miner_uid,
                        external_id,
                        predicted_outcome,
                        predicted_odds,
                        wager,
                        prediction_date,
                    )

        # Batch insert for predictions
        try:
            cls.db_manager.executemany(
                """
                INSERT INTO predictions (
                    prediction_id, game_id, miner_uid, prediction_date, predicted_outcome, 
                    predicted_odds, team_a, team_b, wager, team_a_odds, team_b_odds, 
                    tie_odds, is_model_prediction, outcome, payout, sent_to_site
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prediction_data,
            )
            bt.logging.debug(
                f"Inserted {len(prediction_data)} predictions into the database"
            )
        except Exception as e:
            bt.logging.error(f"Error inserting predictions: {e}")
            raise

        bt.logging.debug("Finished game and prediction simulation")

        return valid_uids, invalid_uids

    def test_combined_scoring_run_with_invalid_uids(self):
        """Combined test for scoring run with approximately 20 invalid UIDs over 60 days."""
        bt.logging.debug("Starting combined scoring run test with invalid UIDs")
        # Introduce approximately 20 invalid UIDs

        # bt.logging.debug(f"Invalid UIDs: {self.invalid_uids}")
        # bt.logging.debug(f"Valid UIDs: {self.valid_uids}")
        valid_uids = self.valid_uids

        # Run scoring for each day
        for day in range(60):
            current_date = self.simulation_start_date + timedelta(days=day)
            date_str = current_date.isoformat()
            bt.logging.debug(f"Running scoring for date: {date_str}")

            weights = self.scoring_system.scoring_run(
                date_str, self.invalid_uids, self.valid_uids
            )

            # Assertions to ensure weights are a numpy array and have the correct shape
            self.assertIsInstance(
                weights, np.ndarray, "Weights should be a NumPy array."
            )
            self.assertEqual(
                weights.shape, (self.num_miners,), "Weights array has incorrect shape."
            )

            # Check that weights sum to 1.0
            self.assertAlmostEqual(
                weights.sum(), 1.0, places=5, msg="Weights do not sum to 1.0."
            )

            # Ensure invalid UIDs have weight 0.0
            for uid in self.invalid_uids:
                self.assertEqual(
                    weights[uid], 0.0, f"Invalid UID {uid} does not have weight 0.0."
                )

            # Log top 5 miners by weight
            top_miners = np.argsort(weights)[-5:][::-1]
            bt.logging.debug(
                f"Top 5 miners for day {day + 1}: {top_miners} with weights {weights[top_miners]}"
            )
    @classmethod
    def verify_game_config(cls, external_id):
        db_result = cls.db_manager.fetch_one(
            "SELECT can_tie FROM game_data WHERE external_id = ?",
            (external_id,)
        )
        db_can_tie = db_result["can_tie"] if db_result else None
        
        entropy_game = cls.scoring_system.entropy_system.game_pools.get(external_id)
        entropy_can_tie = len(entropy_game) == 3 if entropy_game else None
        
        if db_can_tie != entropy_can_tie:
            bt.logging.error(f"Inconsistency for game {external_id}: DB can_tie={db_can_tie}, Entropy can_tie={entropy_can_tie}")
            return False
        return db_can_tie


if __name__ == "__main__":
    unittest.main()