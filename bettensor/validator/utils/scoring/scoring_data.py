import pytz
import bittensor as bt
import torch as t
import sqlite3
from datetime import datetime, timedelta
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
        self.initialize_database()

    def connect_db(self):
        """
        Connect to the SQLite database.

        Returns:
            sqlite3.Connection: SQLite connection object.
        """
        return sqlite3.connect(self.db_path)

    def initialize_database(self):
        """
        Initialize the database by creating necessary tables if they do not exist.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:
            # Create miner_stats table
            cursor.execute(
                """
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
            """
            )

            # Create predictions table
            cursor.execute(
                """
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
            """
            )

            # Create game_data table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS game_data (
                id INTEGER PRIMARY KEY,
                external_id TEXT,
                team_a TEXT,
                team_b TEXT,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                event_start_date TEXT,
                sport TEXT,
                outcome TEXT,
                active INTEGER
            )
            """
            )

            # Create closing_line_odds table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS closing_line_odds (
                game_id INTEGER PRIMARY KEY,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                event_start_date TEXT
            )
            """
            )

            conn.commit()
        except Exception as e:
            bt.logging.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()

    def initialize_miner_stats_table(self):
        """
        Initialize the miner_stats table in the database.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
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
            """
            )
            conn.commit()
        except Exception as e:
            bt.logging.error(f"Error initializing daily stats table: {e}")
            conn.rollback()
        finally:
            conn.close()

    def save_miner_state_file(self, miner_state, miner_state_file):
        """
        Save the miner state to a file.

        Args:
            miner_state: The state of the miner to be saved.
            miner_state_file (str): Path to the file where the state will be saved.
        """
        try:
            with open(miner_state_file, "wb") as f:
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
            with open(miner_state_file, "rb") as f:
                miner_state = t.load(f)
                return miner_state
        except Exception as e:
            bt.logging.error(f"Error loading miner state file: {e}")
            return None

    def initialize_new_miner(self, miner_uid: int):
        """
        Initialize a new miner in the database.

        Args:
            miner_uid (int): Unique identifier for the miner.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM miner_stats WHERE miner_uid = ?
            """,
                (miner_uid,),
            )
            if cursor.fetchone():
                bt.logging.info(
                    f"Miner {miner_uid} already exists in the database. Updating..."
                )
                cursor.execute(
                    """
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
                """,
                    (None, None, miner_uid),
                )
            else:
                cursor.execute(
                    """
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
                """,
                    (miner_uid, None, None),
                )
            conn.commit()
        except Exception as e:
            bt.logging.error(f"Error initializing new miner: {e}")
            conn.rollback()
        finally:
            conn.close()

    def update_miner_stats(self, miner_stats: Dict[int, Dict]):
        """
        Update the statistics of miners in the database.

        Args:
            miner_stats (Dict[int, Dict]): Dictionary containing miner statistics.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:
            for miner_uid, stats in miner_stats.items():
                cursor.execute(
                    """
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
                """,
                    (
                        stats["status"],
                        stats["cash"],
                        stats["current_incentive"],
                        stats["current_tier"],
                        stats["current_scoring_window"],
                        stats["current_composite_score"],
                        stats["current_sharpe_ratio"],
                        stats["current_sortino_ratio"],
                        stats["current_roi"],
                        stats["current_clv_avg"],
                        stats["last_prediction_date"],
                        stats["earnings"],
                        stats["wager_amount"],
                        stats["profit"],
                        stats["predictions"],
                        stats["wins"],
                        stats["losses"],
                        stats["wins"],
                        stats["wins"],
                        stats["losses"],
                        str(miner_uid),  # Convert miner_uid to string
                    ),
                )

            conn.commit()
            bt.logging.info(
                f"Successfully updated miner stats for {len(miner_stats)} miners."
            )
        except Exception as e:
            bt.logging.error(f"Error updating miner stats: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_closed_predictions_for_day(self, date: str):
        """
        Get closed predictions for a specific day.

        Args:
            date (str): The date for which to retrieve closed predictions.

        Returns:
            List of closed predictions or None if an error occurs.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM predictions WHERE prediction_date = ? AND game_status != 'Unfinished'
            """,
                (date,),
            )
            predictions = cursor.fetchall()
            return predictions
        except Exception as e:
            bt.logging.error(f"Error getting predictions for day: {e}")
            return None

    def get_closed_games_for_day(self, date: str):
        """
        Get closed games for a specific day.

        Args:
            date (str): The date for which to retrieve closed games.

        Returns:
            List of closed games or None if an error occurs.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM games WHERE game_date = ? AND game_status != 'Unfinished'
            """,
                (date,),
            )
            games = cursor.fetchall()
            return games
        except Exception as e:
            bt.logging.error(f"Error getting games for day: {e}")
            return None

    def ensure_closing_line_odds_table(self):
        """
        Ensure the closing_line_odds table exists in the database.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS closing_line_odds (
                    game_id INTEGER PRIMARY KEY,
                    team_a_odds REAL,
                    team_b_odds REAL,
                    tie_odds REAL,
                    event_start_date TEXT
                )
            """
            )
            conn.commit()
        except Exception as e:
            bt.logging.error(f"Error ensuring closing_line_odds table: {e}")
            conn.rollback()
        finally:
            conn.close()

    def fetch_closing_line_odds(self):
        """
        Fetch closing line odds for games starting within 15 minutes and insert them into the database.
        """
        try:
            conn = self.connect_db()
            cursor = conn.cursor()

            # Get current time (UTC)
            current_time = datetime.now(pytz.utc)

            # find games that are starting within 15 minutes
            cursor.execute(
                """
                SELECT id, team_a_odds, team_b_odds, tie_odds, event_start_date
                FROM game_data
                WHERE event_start_date BETWEEN ? AND ?
            """,
                (
                    current_time - timedelta(minutes=15),
                    current_time + timedelta(minutes=15),
                ),
            )
            games = cursor.fetchall()

            # insert the games into the closing_line_odds table
            for game in games:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO closing_line_odds (game_id, team_a_odds, team_b_odds, tie_odds, event_start_date)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    game,
                )
            conn.commit()
        except Exception as e:
            bt.logging.error(f"Error fetching closing line odds: {e}")
            conn.rollback()
        finally:
            conn.close()

    def preprocess_for_scoring(self, date):
        """
        Preprocess data for scoring.

        Args:
            date (str): The date for which to preprocess data.

        Returns:
            Tuple containing predictions, closing line odds, and results.
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        games = self._fetch_game_data(cursor, date)
        if not games:
            bt.logging.warning(
                f"No active games found for date {date}. Checking database content..."
            )
            cursor.execute("SELECT COUNT(*) FROM game_data")
            total_games = cursor.fetchone()[0]
            cursor.execute("SELECT DISTINCT event_start_date FROM game_data")
            available_dates = [row[0] for row in cursor.fetchall()]
            bt.logging.warning(f"Total games in database: {total_games}")
            bt.logging.warning(f"Available dates: {available_dates}")
            return [], t.empty((0, 4)), t.empty(0)

        bt.logging.info(f"Found {len(games)} games for date {date}")
        game_id_map = {
            game[0]: i for i, game in enumerate(games)
        }  # game[0] is now external_id
        closing_line_odds = t.tensor(
            [[i] + [float(odds) for odds in game[1:4]] for i, game in enumerate(games)],
            dtype=t.float32,
        )
        results = t.tensor([float(game[4]) for game in games], dtype=t.float32)

        predictions_data = self._fetch_predictions(cursor, date)
        conn.close()

        if not predictions_data:
            bt.logging.warning(f"No predictions found for date {date}")
            return [], closing_line_odds, results

        bt.logging.info(f"Found {len(predictions_data)} predictions for date {date}")
        predictions = [[] for _ in range(self.num_miners)]
        skipped_predictions = 0
        for miner_uid, game_id, outcome, odds, wager in predictions_data:
            miner_uid = int(miner_uid)
            if game_id in game_id_map:
                predictions[miner_uid].append(
                    [game_id_map[game_id], float(outcome), float(odds), float(wager)]
                )
            else:
                bt.logging.warning(
                    f"Prediction for non-existent game {game_id} encountered. Skipping this prediction."
                )
                skipped_predictions += 1

        predictions = [
            t.tensor(preds, dtype=t.float32) if preds else t.empty((0, 4))
            for preds in predictions
        ]

        valid_predictions = sum(len(p) for p in predictions)
        bt.logging.info(f"Processed {valid_predictions} valid predictions")
        bt.logging.info(f"Skipped {skipped_predictions} invalid predictions")
        return predictions, closing_line_odds, results

    def _fetch_game_data(self, cursor, date):
        """
        Fetch game data for a specific date.

        Args:
            cursor (sqlite3.Cursor): SQLite cursor object.
            date (str): The date for which to fetch game data.

        Returns:
            List of game data tuples.
        """
        cursor.execute(
            """
            SELECT external_id, team_a_odds, team_b_odds, tie_odds, outcome
            FROM game_data
            WHERE DATE(event_start_date) = DATE(?)
        """,
            (date,),
        )
        return cursor.fetchall()

    def _fetch_predictions(self, cursor, date):
        """
        Fetch predictions for a specific date.

        Args:
            cursor (sqlite3.Cursor): SQLite cursor object.
            date (str): The date for which to fetch predictions.

        Returns:
            List of prediction data tuples.
        """
        cursor.execute(
            """
            SELECT miner_uid, game_id, predicted_outcome, predicted_odds, wager
            FROM predictions
            WHERE DATE(prediction_date) = DATE(?)
        """,
            (date,),
        )
        return cursor.fetchall()

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
            if not all(
                isinstance(x, (int, float))
                for x in [team_a_odds, team_b_odds, tie_odds]
            ):
                bt.logging.warning(f"Invalid odds for game {game_id}. Skipping.")
                continue

            if outcome not in [
                0,
                1,
                2,
            ]:  # Assuming 0: team_a win, 1: team_b win, 2: tie
                bt.logging.warning(f"Invalid outcome for game {game_id}. Skipping.")
                continue

            games_dict[game_id] = (team_a_odds, team_b_odds, tie_odds, outcome)

        return games_dict

    def _process_prediction_data(
        self, predictions_data: List[Tuple], games_dict: Dict[int, Tuple]
    ) -> Dict[int, List[Tuple]]:
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
                bt.logging.warning(
                    f"Prediction for non-existent game {game_id}. Skipping."
                )
                continue

            if predicted_outcome not in [0, 1, 2]:
                bt.logging.warning(
                    f"Invalid predicted outcome for game {game_id}. Skipping."
                )
                continue

            if not isinstance(predicted_odds, (int, float)) or predicted_odds <= 1:
                bt.logging.warning(
                    f"Invalid predicted odds for game {game_id}. Skipping."
                )
                continue

            if not isinstance(wager, (int, float)) or wager <= 0:
                bt.logging.warning(f"Invalid wager for game {game_id}. Skipping.")
                continue

            if miner_uid not in miner_predictions:
                miner_predictions[miner_uid] = []
            miner_predictions[miner_uid].append(
                (game_id, predicted_outcome, predicted_odds, wager)
            )

        return miner_predictions

    def _prepare_tensors(
        self, miner_predictions: Dict[int, List[Tuple]], games_dict: Dict[int, Tuple]
    ) -> Tuple[List[t.Tensor], t.Tensor, t.Tensor]:
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
            miner_tensor = t.tensor(
                [
                    [game_id, predicted_outcome, predicted_odds, wager]
                    for game_id, predicted_outcome, predicted_odds, wager in preds
                ]
            )
            predictions.append(miner_tensor)

        closing_line_odds = t.tensor(
            [
                [game_id, game_data[0], game_data[1], game_data[2]]
                for game_id, game_data in games_dict.items()
            ]
        )
        results = t.tensor([game_data[3] for game_data in games_dict.values()])

        return predictions, closing_line_odds, results
