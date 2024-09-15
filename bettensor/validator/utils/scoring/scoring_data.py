    
import pytz
import bettensor as bt
import torch as t
import sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

#### MINER STATS ####

def initialize_miner_stats_table(self):
        '''
        initialize the miner stats table in the database

        
        '''
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS miner_stats (
                miner_hotkey TEXT PRIMARY KEY,
                miner_coldkey TEXT,
                miner_uid TEXT,
                miner_rank INTEGER,
                miner_status TEXT,
                miner_cash REAL,
                miner_current_incentive REAL,
                miner_current_tier INTEGER,
                miner_current_scoring_window INTEGER,
                miner_current_composite_score REAL,
                miner_current_sharpe_ratio REAL,
                miner_current_sortino_ratio REAL,
                miner_current_roi REAL,
                miner_current_clv_avg REAL,
                miner_last_prediction_date TEXT,
                miner_lifetime_earnings REAL,
                miner_lifetime_wager_amount REAL,
                miner_lifetime_profit REAL,
                miner_lifetime_predictions INTEGER,
                miner_lifetime_wins INTEGER,
                miner_lifetime_losses INTEGER,
                miner_win_loss_ratio REAL
            )
            """)

        except Exception as e:
            bt.logging.error(f"Error initializing daily stats table: {e}")
            conn.rollback()
        finally:
            conn.close()

def save_miner_state_file(self):
    '''
    Saves the current miner state to a torch file, in case of restarts/etc. Takes state from the most recent scoring run. (i.e., this is called after every scoring run)

    '''

    try:    
        with open(self.miner_state_file, 'wb') as f:
            t.save(self.miner_state, f)
    except Exception as e:
        bt.logging.error(f"Error saving miner state file: {e}") 

def load_miner_state_file(self):
    '''
    Loads the most recent miner state from a torch file, in case of restarts/etc

    '''

    try:
        with open(self.miner_state_file, 'rb') as f:
            miner_state = t.load(f)
            return miner_state
    except Exception as e:
        bt.logging.error(f"Error loading miner state file: {e}")
        return None

def initialize_new_miner(self, miner_uid: int):
    '''
    Initializes a new miner in the database, replacing the old uid if it exists. This method should be called whenever a new miner is detected in the validator loop.
    '''
    conn = self.connect_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM miner_stats WHERE miner_uid = ?
        """, (miner_uid,))
        if cursor.fetchone():
            bt.logging.info(f"Miner {miner_uid} already exists in the database. Updating...")
            cursor.execute("""
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
            """, (None, None, miner_uid))  # Assuming hotkey and coldkey are initially None
        else:
            cursor.execute("""
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
            """, (miner_uid, None, None))
        conn.commit()
    except Exception as e:
        bt.logging.error(f"Error initializing new miner: {e}")
        conn.rollback()
    finally:
        conn.close()

def update_miner_stats(self, miner_stats: Dict[int, Dict]):
    '''
    Updates the miner stats table in the database with the given miner stats, after a scoring run.
    
    Args:
    miner_stats (Dict[int, Dict]): A dictionary with miner UIDs as keys and their updated stats as values.
    '''
    conn = self.connect_db()
    cursor = conn.cursor()

    try:
        for miner_uid, stats in miner_stats.items():
            cursor.execute("""
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
            """, (
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
                miner_uid
            ))
        
        conn.commit()
        bt.logging.info(f"Successfully updated miner stats for {len(miner_stats)} miners.")
    except Exception as e:
        bt.logging.error(f"Error updating miner stats: {e}")
        conn.rollback()
    finally:
        conn.close()





#### PREDICTIONS ####
def get_closed_predictions_for_day(self, date: str):
    '''
    Gets all predictions for a given day, for games that have finished.
    '''
    conn = self.connect_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM predictions WHERE prediction_date = ? AND game_status != 'Unfinished'
        """, (date,))
        predictions = cursor.fetchall()
        return predictions
    except Exception as e:
        bt.logging.error(f"Error getting predictions for day: {e}")
        return None

def get_closed_games_for_day(self, date: str):
    '''
    Gets all games for a given day, for games that have finished.
    '''
    conn = self.connect_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT * FROM games WHERE game_date = ? AND game_status != 'Unfinished'
        """, (date,))
        games = cursor.fetchall()
        return games
    except Exception as e:
        bt.logging.error(f"Error getting games for day: {e}")
        return None
    

#### CLOSING LINE ODDS ####


def ensure_closing_line_odds_table(self):
    '''
    Ensure the closing_line_odds table exists in the database.
    '''
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS closing_line_odds (
                game_id INTEGER PRIMARY KEY,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                event_start_date TEXT
            )
        """)
        conn.commit()
    except Exception as e:
        bt.logging.error(f"Error ensuring closing_line_odds table: {e}")
        conn.rollback()
    finally:
        conn.close()

def fetch_closing_line_odds(self):
    '''
    Find games in the database that are starting within 15 minutes. Record those odds as the closing line odds. Store them in a table. This method should be run periodically in the validator loop.
    '''
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current time (UTC)
        current_time = datetime.now(pytz.utc)

        #find games that are starting within 15 minutes
        cursor.execute("""
            SELECT id, team_a_odds, team_b_odds, tie_odds, event_start_date
            FROM game_data
            WHERE event_start_date BETWEEN ? AND ?
        """, (current_time - timedelta(minutes=15), current_time + timedelta(minutes=15)))
        games = cursor.fetchall()

        #insert the games into the closing_line_odds table
        for game in games:
            cursor.execute("""
                INSERT OR REPLACE INTO closing_line_odds (game_id, team_a_odds, team_b_odds, tie_odds, event_start_date)
                VALUES (?, ?, ?, ?, ?)
            """, game)
        conn.commit()
    except Exception as e:
        bt.logging.error(f"Error fetching closing line odds: {e}")
        conn.rollback()
    finally:
        conn.close()

    
## Entropy System ##




### RUN ALL PREPROCESSING FUNCTIONS ####

def preprocess_for_scoring(self, date: str) -> Tuple[List[t.Tensor], t.Tensor, t.Tensor]:
    '''
    Preprocesses all data needed for scoring. Takes a date as input and returns the necessary data for scoring.
    '''
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Validate input date
        try:
            scoring_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got {date}")

        # Fetch predictions for the given date
        cursor.execute("""
            SELECT miner_uid, game_id, predicted_outcome, predicted_odds, wager
            FROM predictions
            WHERE DATE(prediction_date) = DATE(?)
        """, (date,))
        predictions_data = cursor.fetchall()

        # Fetch games that have finished
        cursor.execute("""
            SELECT id, team_a_odds, team_b_odds, tie_odds, outcome
            FROM game_data
            WHERE DATE(event_start_date) = DATE(?) AND outcome IS NOT NULL
        """, (date,))
        games_data = cursor.fetchall()

        conn.close()

        # Validate fetched data
        if not predictions_data:
            bt.logging.warning(f"No predictions found for date {date}")
        if not games_data:
            bt.logging.warning(f"No finished games found for date {date}")

        # Process and validate game data
        games_dict = self._process_game_data(games_data)

        # Process and validate prediction data
        miner_predictions = self._process_prediction_data(predictions_data, games_dict)

        # Prepare data in the format expected by ScoringSystem
        predictions, closing_line_odds, results = self._prepare_tensors(miner_predictions, games_dict)

        return predictions, closing_line_odds, results

    except Exception as e:
        bt.logging.error(f"Error in preprocess_for_scoring: {str(e)}")
        raise

def _process_game_data(self, games_data: List[Tuple]) -> Dict[int, Tuple]:
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
    predictions = []
    for miner_uid, preds in miner_predictions.items():
        miner_tensor = t.tensor([[game_id, predicted_outcome, predicted_odds] for game_id, predicted_outcome, predicted_odds, _ in preds])
        predictions.append(miner_tensor)

    closing_line_odds = t.tensor([[game_id, game_data[0]] for game_id, game_data in games_dict.items()])
    results = t.tensor([game_data[3] for game_data in games_dict.values()])

    return predictions, closing_line_odds, results

