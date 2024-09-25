    
import pytz
import bittensor as bt
import torch as t
from datetime import datetime, timedelta
from bettensor.validator.utils.database.database_manager import DatabaseManager
from typing import List, Tuple, Dict

class ScoringData:
    def __init__(self, db_path: str, num_miners: int):
        """
        Initialize the ScoringData object.

        Args:
            db_path (str): Path to the SQLite database file.
            num_miners (int): Number of miners.
        """
        self.db_path = db_path
        self.num_miners = num_miners
        self.db_manager = DatabaseManager(db_path)

    def save_miner_state_file(self, miner_state, miner_state_file):
        """
        Save the miner state to a file.

        Args:
            miner_state: The state of the miner to be saved.
            miner_state_file (str): Path to the file where the state will be saved.
        """
        try:    
            with open(miner_state_file, 'wb') as f:
                t.save(miner_state, f)
        except Exception as e:
            bt.logging.error(f"Error saving miner state file: {e}") 

    def load_miner_state_file(self, miner_state_file):
        """
        Load the miner state from a file.

        Args:
            miner_state_file (str): Path to the file from which the state will be loaded.

        Returns:
            The loaded miner state or None if an error occurs.
        """
        try:
            with open(miner_state_file, 'rb') as f:
                miner_state = t.load(f)
                return miner_state
        except Exception as e:
            bt.logging.error(f"Error loading miner state file: {e}")
            return None

    def initialize_new_miner(self, miner_uid: int):
        try:
            query = "SELECT * FROM miner_stats WHERE miner_uid = ?"
            existing_miner = self.db_manager.fetch_one(query, (miner_uid,))
            
            if existing_miner:
                bt.logging.info(f"Miner {miner_uid} already exists in the database. Updating...")
                update_query = """
                    UPDATE miner_stats          
                    SET miner_hotkey = ?,
                        miner_coldkey = ?,
                        miner_status = 'active',
                        miner_cash = 0,
                        miner_current_incentive = 0,
                        miner_current_tier = 0,
                        miner_current_scoring_window = 0,
                        miner_current_composite_score = 0,
                        miner_current_sharpe_ratio = 0,
                        miner_current_sortino_ratio = 0,
                        miner_current_roi = 0,
                        miner_current_clv_avg = 0,
                        miner_last_prediction_date = '0',
                        miner_lifetime_earnings = 0,
                        miner_lifetime_wager_amount = 0,
                        miner_lifetime_profit = 0,
                        miner_lifetime_predictions = 0,
                        miner_lifetime_wins = 0,
                        miner_lifetime_losses = 0,
                        miner_win_loss_ratio = 0 
                    WHERE miner_uid = ?
                """
                self.db_manager.execute_query(update_query, (None, None, miner_uid))
            else:
                insert_query = """
                    INSERT INTO miner_stats (
                        miner_uid, miner_hotkey, miner_coldkey, miner_status, miner_cash,
                        miner_current_incentive, miner_current_tier, miner_current_scoring_window,
                        miner_current_composite_score, miner_current_sharpe_ratio,
                        miner_current_sortino_ratio, miner_current_roi, miner_current_clv_avg,
                        miner_last_prediction_date, miner_lifetime_earnings,
                        miner_lifetime_wager_amount, miner_lifetime_profit,
                        miner_lifetime_predictions, miner_lifetime_wins,
                        miner_lifetime_losses, miner_win_loss_ratio
                    ) VALUES (?, ?, ?, 'active', 0, 0, 0, 0, 0, 0, 0, 0, 0, '0', 0, 0, 0, 0, 0, 0, 0)
                """
                self.db_manager.execute_query(insert_query, (miner_uid, None, None))
        except Exception as e:
            bt.logging.error(f"Error initializing new miner: {e}")

    def update_miner_stats(self, miner_stats: Dict[int, Dict]):
        try:
            update_query = """
                UPDATE miner_stats
                SET miner_status = ?,
                    miner_cash = ?,
                    miner_current_incentive = ?,
                    miner_current_tier = ?,
                    miner_current_scoring_window = ?,
                    miner_current_composite_score = ?,
                    miner_current_sharpe_ratio = ?,
                    miner_current_sortino_ratio = ?,
                    miner_current_roi = ?,
                    miner_current_clv_avg = ?,
                    miner_last_prediction_date = ?,
                    miner_lifetime_earnings = miner_lifetime_earnings + ?,
                    miner_lifetime_wager_amount = miner_lifetime_wager_amount + ?,
                    miner_lifetime_profit = miner_lifetime_profit + ?,
                    miner_lifetime_predictions = miner_lifetime_predictions + ?,
                    miner_lifetime_wins = miner_lifetime_wins + ?,
                    miner_lifetime_losses = miner_lifetime_losses + ?,
                    miner_win_loss_ratio = CASE 
                        WHEN (miner_lifetime_wins + ?) > 0 THEN 
                            (miner_lifetime_wins + ?) * 1.0 / NULLIF((miner_lifetime_losses + ?), 0)
                        ELSE 0 
                    END
                WHERE miner_uid = ?
            """
            batch_params = [
                (
                    stats['status'],
                    stats['cash'],
                    stats['current_incentive'],
                    stats['current_tier'],
                    stats['current_scoring_window'],
                    stats['current_composite_score'],
                    stats['current_sharpe_ratio'],
                    stats['current_sortino_ratio'],
                    stats['current_roi'],
                    stats['current_clv_avg'],
                    stats['last_prediction_date'],
                    stats['earnings'],
                    stats['wager_amount'],
                    stats['profit'],
                    stats['predictions'],
                    stats['wins'],
                    stats['losses'],
                    stats['wins'],
                    stats['wins'],
                    stats['losses'],
                    str(miner_uid)
                )
                for miner_uid, stats in miner_stats.items()
            ]
            self.db_manager.execute_query(update_query, batch_params, batch=True)
            bt.logging.info(f"Successfully updated miner stats for {len(miner_stats)} miners.")
        except Exception as e:
            bt.logging.error(f"Error updating miner stats: {e}")

    def get_closed_predictions_for_day(self, date: str):
        try:
            query = """
                SELECT * FROM predictions WHERE prediction_date = ? AND outcome != 'Unfinished'
            """
            return self.db_manager.fetch_all(query, (date,))
        except Exception as e:
            bt.logging.error(f"Error getting predictions for day: {e}")
            return None

    def get_closed_games_for_day(self, date: str):
        try:
            query = """
                SELECT * FROM game_data WHERE event_start_date = ? AND outcome != 'Unfinished'
            """
            return self.db_manager.fetch_all(query, (date,))
        except Exception as e:
            bt.logging.error(f"Error getting games for day: {e}")
            return None

    def preprocess_for_scoring(self, date: str):
        """
        Preprocess data for scoring.

        Args:
            date (str): The date for which to preprocess data.

        Returns:
            Tuple containing predictions and results.
        """
        try:
            games = self._fetch_closed_game_data(date)
            if not games:
                bt.logging.warning(f"No closed games found for date {date}. Checking database content...")
                total_games = self.db_manager.fetch_one("SELECT COUNT(*) FROM game_data", ())[0]
                available_dates = self.db_manager.fetch_all("SELECT DISTINCT event_start_date FROM game_data", ())
                available_dates = [row[0] for row in available_dates]
                bt.logging.warning(f"Total games in database: {total_games}")
                bt.logging.warning(f"Available dates: {available_dates}")
                return [], t.empty((0, 4)), t.empty(0)

            bt.logging.info(f"Found {len(games)} closed games for date {date}")
            games_dict = self._process_game_data(games)
            results = t.tensor([game_data[3] for game_data in games_dict.values()], dtype=t.float32)

            predictions_data = self._fetch_predictions(date)

            predictions = [[] for _ in range(self.num_miners)]
            skipped_predictions = 0
            for miner_uid, game_id, outcome, odds, wager in predictions_data:
                miner_uid = int(miner_uid)
                if game_id in games_dict:
                    predictions[miner_uid].append([game_id, float(outcome), float(odds), float(wager)])
                else:
                    bt.logging.warning(f"Prediction for non-existent game {game_id} encountered. Skipping this prediction.")
                    skipped_predictions += 1

            predictions = [t.tensor(preds, dtype=t.float32) if preds else t.empty((0, 4)) for preds in predictions]

            valid_predictions = sum(len(p) for p in predictions)
            bt.logging.info(f"Processed {valid_predictions} valid predictions")
            bt.logging.info(f"Skipped {skipped_predictions} invalid predictions")
            bt.logging.info(f"Predictions length: {len(predictions)}")
            return predictions, results
        except Exception as e:
            bt.logging.error(f"Error in preprocess_for_scoring: {e}")
            return [], t.empty((0, 4)), t.empty(0)

    def _fetch_closed_game_data(self, date):
        query = """
        SELECT external_id, team_a_odds, team_b_odds, tie_odds, outcome
        FROM game_data
        WHERE event_start_date = ? AND outcome != 'Unfinished'
        """
        return self.db_manager.fetch_all(query, (date,))

    def _fetch_predictions(self, date):
        query = """
        SELECT p.miner_uid, p.game_id, p.predicted_outcome, p.predicted_odds, p.wager
        FROM predictions p
        JOIN game_data g ON p.game_id = g.external_id
        WHERE g.event_start_date = ? AND g.active = 1
        """
        return self.db_manager.fetch_all(query, (date,))

    def _process_game_data(self, games_data: List[Tuple]) -> Dict[int, Tuple]:
        """
        Process game data into a dictionary.

        Args:
            games_data (List[Tuple]): List of game data tuples.

        Returns:
            Dictionary of processed game data.
        """
        games_dict = {}
        for game in games_data:
            game_id, team_a_odds, team_b_odds, tie_odds, outcome = game
            
            # Validate game data
            if not all(isinstance(x, (int, float)) for x in [team_a_odds, team_b_odds, tie_odds]):
                bt.logging.warning(f"Invalid odds for game {game_id}. Skipping.")
                continue
            
            if outcome not in [0, 1, 2]:  # Assuming 0: team_a win, 1: team_b win, 2: tie
                bt.logging.warning(f"Invalid outcome for game {game_id}. Skipping.")
                continue
            
            games_dict[game_id] = (team_a_odds, team_b_odds, tie_odds, outcome)
        
        return games_dict

    def _process_prediction_data(self, predictions_data: List[Tuple], games_dict: Dict[int, Tuple]) -> Dict[int, List[Tuple]]:
        """
        Process prediction data into a dictionary.

        Args:
            predictions_data (List[Tuple]): List of prediction data tuples.
            games_dict (Dict[int, Tuple]): Dictionary of game data.

        Returns:
            Dictionary of processed prediction data.
        """
        miner_predictions = {}
        for pred in predictions_data:
            miner_uid, game_id, predicted_outcome, predicted_odds, wager = pred
            
            # Validate prediction data
            if game_id not in games_dict:
                bt.logging.warning(f"Prediction for non-existent game {game_id}. Skipping.")
                continue
            
            if predicted_outcome not in [0, 1, 2]:
                bt.logging.warning(f"Invalid predicted outcome for game {game_id}. Skipping.")
                continue
            
            if not isinstance(predicted_odds, (int, float)) or predicted_odds <= 1:
                bt.logging.warning(f"Invalid predicted odds for game {game_id}. Skipping.")
                continue
            
            if not isinstance(wager, (int, float)) or wager <= 0:
                bt.logging.warning(f"Invalid wager for game {game_id}. Skipping.")
                continue
            
            if miner_uid not in miner_predictions:
                miner_predictions[miner_uid] = []
            miner_predictions[miner_uid].append((game_id, predicted_outcome, predicted_odds, wager))
        
        return miner_predictions

    def _prepare_tensors(self, miner_predictions: Dict[int, List[Tuple]], games_dict: Dict[int, Tuple]) -> Tuple[List[t.Tensor], t.Tensor, t.Tensor]:
        """
        Prepare tensors for predictions, closing line odds, and results.

        Args:
            miner_predictions (Dict[int, List[Tuple]]): Dictionary of miner predictions.
            games_dict (Dict[int, Tuple]): Dictionary of game data.

        Returns:
            Tuple containing lists of tensors for predictions, closing line odds, and results.
        """
        predictions = []
        for uid, preds in miner_predictions.items():
            miner_tensor = t.tensor([[game_id, predicted_outcome, predicted_odds, wager] for game_id, predicted_outcome, predicted_odds, wager in preds])
            predictions.append(miner_tensor)
        bt.logging.info(f"Prepared {len(predictions)} predictions tensors")
        closing_line_odds = t.tensor([[game_id, game_data[0], game_data[1], game_data[2]] for game_id, game_data in games_dict.items()])
        results = t.tensor([game_data[3] for game_data in games_dict.values()])

        return predictions, closing_line_odds, results

    def process_predictions(self, date: str):
        """
        Process predictions for closed games on a given date.

        Args:
            date (str): The date for which to process predictions.
        """
        try:
            games = self._fetch_closed_game_data(date)
            if not games:
                bt.logging.warning(f"No closed games found for date {date}. No predictions to process.")
                return

            games_dict = self._process_game_data(games)
            for game_id, game_data in games_dict.items():
                outcome = game_data[3]  # Assuming outcome is at index 3
                predictions = self._fetch_predictions_for_game(game_id)
                
                for prediction in predictions:
                    miner_uid, pred_game_id, predicted_outcome, predicted_odds, wager = prediction
                    is_winner = predicted_outcome == outcome
                    payout = wager * predicted_odds if is_winner else 0

                    update_query = """
                        UPDATE predictions
                        SET result = ?, payout = ?
                        WHERE miner_uid = ? AND game_id = ?
                    """
                    self.db_manager.execute_query(update_query, (is_winner, payout, miner_uid, pred_game_id))
                    
                    if is_winner:
                        bt.logging.info(f"Miner {miner_uid} won prediction on game {game_id}. Payout: {payout}")
                    else:
                        bt.logging.info(f"Miner {miner_uid} lost prediction on game {game_id}.")
            
            bt.logging.info(f"Processed predictions for all closed games on date {date}.")
        except Exception as e:
            bt.logging.error(f"Error in process_predictions: {e}")

    def _fetch_predictions_for_game(self, game_id: int):
        """
        Fetch all predictions for a specific game.

        Args:
            game_id (int): The external_id of the game.

        Returns:
            List of predictions tuples.
        """
        try:
            query = """
                SELECT miner_uid, game_id, predicted_outcome, predicted_odds, wager
                FROM predictions
                WHERE game_id = ? AND processed = 0
            """
            return self.db_manager.fetch_all(query, (game_id,))
        except Exception as e:
            bt.logging.error(f"Error fetching predictions for game {game_id}: {e}")
            return []

