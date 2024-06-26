import schedule
import datetime
import sqlite3
import bittensor as bt
import time
from datetime import datetime, timedelta
from bettensor.protocol import MinerStats
import threading
import pytz

import datetime
import pytz

class MinerStatsHandler:
    """
    This class is used to store miner stats and perform calculations on the data. It is instantiated for each validator and miner.

    """

    def __init__(self, db_path, profile):
        self.db_path = db_path
        self.profile = profile
        self.conn = self.connect_to_db(self.db_path)
        self.create_table(self.conn)

        # Start the run method in a separate thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def connect_to_db(self, db_path):
        """
        This method connects to the database.
        """
        conn = sqlite3.connect(db_path)
        return conn

    def create_table(self, conn):
        """
        This method creates a table in the database to store miner stats.

        fields:
        miner_hotkey :                 hotkey of the miner
        miner_coldkey :                coldkey of the miner
        miner_uid :                    current uid of the miner
        miner_rank :                   current rank of the miner
        miner_cash :                   current cash of the miner
        miner_current_incentive :      current incentive of the miner
        miner_last_prediction_date :   date of the last prediction of the miner
        miner_lifetime_earnings :      lifetime earnings of the miner
        miner_lifetime_wager :         lifetime wager of the miner
        miner_lifetime_predictions :   lifetime predictions of the miner
        miner_lifetime_wins :          lifetime wins of the miner
        miner_lifetime_losses :        lifetime losses of the miner
        miner_win_loss_ratio :         win loss ratio of the miner
        miner_status :                 status of the miner (active, inactive, banned, deregistered)
        """
        c = conn.cursor()
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS miner_stats (
            miner_hotkey TEXT PRIMARY KEY,
            miner_coldkey TEXT,
            miner_uid INTEGER,
            miner_rank INTEGER,
            miner_cash REAL,
            miner_current_incentive REAL,
            miner_last_prediction_date TEXT,
            miner_lifetime_earnings REAL,
            miner_lifetime_wager REAL,
            miner_lifetime_predictions INTEGER,
            miner_lifetime_wins INTEGER,
            miner_lifetime_losses INTEGER,
            miner_win_loss_ratio REAL,
            miner_status TEXT
        )"""
        )
        conn.commit()

    def reset_daily_cash(self):
        """
        This method resets the daily cash of every miner to 1000, executed at 00:00 UTC
        """
        with self.connect_to_db(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
            UPDATE miner_stats
            SET miner_cash = 1000
            """
            )
            conn.commit()
        bt.logging.info("Daily cash reset for all miners")

        # TODO: trigger miner_stats update query

    def init_miner_row(self, miner_hotkey: str, miner_uid: int) -> bool:
        """
        This method is called when the validator discovers a new miner on the network, or on validator startup for each hotkey, or when the miner starts up.
        It inserts the miner's hotkey, coldkey, uid, rank, cash, current incentive, last prediction date, lifetime earnings, lifetime wager,
        lifetime predictions, lifetime wins, lifetime losses, win loss ratio, and status into the database.
        """
        miner_hotkey = miner_hotkey

        miner_uid = miner_uid

        # init other values to 0
        miner_coldkey = ""
        miner_rank = 0  # rank 0 is init value before rank is assigned
        miner_cash = 1000
        miner_current_incentive = 0
        miner_last_prediction_date = ""
        miner_lifetime_earnings = 0
        miner_lifetime_wager = 0
        miner_lifetime_predictions = 0
        miner_lifetime_wins = 0
        miner_lifetime_losses = 0
        miner_win_loss_ratio = 0
        miner_status = "active"

        c = self.conn.cursor()
        # check if miner_hotkey already exists
        c.execute(
            """
        SELECT * FROM miner_stats
        WHERE miner_hotkey = ?
        """,
            (miner_hotkey,),
        )
        if c.fetchone():
            return True
        else:
            pass

        c.execute(
            """
        INSERT INTO miner_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                miner_hotkey,
                miner_coldkey,
                miner_uid,
                miner_rank,
                miner_cash,
                miner_current_incentive,
                miner_last_prediction_date,
                miner_lifetime_earnings,
                miner_lifetime_wager,
                miner_lifetime_predictions,
                miner_lifetime_wins,
                miner_lifetime_losses,
                miner_win_loss_ratio,
                miner_status,
            ),
        )
        self.conn.commit()

        return True

    def update_miner_row(self, conn, miner: MinerStats) -> bool:
        """
        This method is called on a miner row when some event triggers it (new game outcome, miner makes a prediction, etc)
        Args:
            conn : sqlite3 connection
            miner : MinerStats object
        Returns:
            bool : True if the miner row was updated, False otherwise
        """
        c = conn.cursor()
        miner_hotkey = miner.miner_hotkey
        miner_coldkey = miner.miner_coldkey
        miner_uid = miner.miner_uid
        miner_rank = miner.miner_rank
        miner_cash = miner.miner_cash
        miner_current_incentive = miner.miner_current_incentive
        miner_last_prediction_date = miner.miner_last_prediction_date
        miner_lifetime_earnings = miner.miner_lifetime_earnings
        miner_lifetime_wager = miner.miner_lifetime_wager
        miner_lifetime_predictions = miner.miner_lifetime_predictions
        miner_lifetime_wins = miner.miner_lifetime_wins
        miner_lifetime_losses = miner.miner_lifetime_losses
        miner_win_loss_ratio = miner.miner_win_loss_ratio
        miner_status = miner.miner_status

        c.execute(
            """
        UPDATE miner_stats SET 
        miner_uid = ?,
        miner_rank = ?,
        miner_cash = ?,
        miner_current_incentive = ?,
        miner_last_prediction_date = ?,
        miner_lifetime_earnings = ?,
        miner_lifetime_wager = ?,
        miner_lifetime_predictions = ?,
        miner_lifetime_wins = ?,
        miner_lifetime_losses = ?,
        miner_win_loss_ratio = ?,
        miner_status = ?
        WHERE miner_hotkey = ?
        """,
            (
                miner_uid,
                miner_rank,
                miner_cash,
                miner_current_incentive,
                miner_last_prediction_date,
                miner_lifetime_earnings,
                miner_lifetime_wager,
                miner_lifetime_predictions,
                miner_lifetime_wins,
                miner_lifetime_losses,
                miner_win_loss_ratio,
                miner_status,
                miner_hotkey,
            ),
        )
        conn.commit()

        return True

    def return_miner_stats(self, conn, miner_hotkey: str) -> MinerStats:
        """
        This method returns the miner row from the database as a MinerStats object, using the create method from the MinerStats class
        """
        c = conn.cursor()
        c.execute(
            """
        SELECT * FROM miner_stats
        WHERE miner_hotkey = ?
        """,
            (miner_hotkey,),
        )
        row = c.fetchone()

        return MinerStats.create(row)

    def stop(self):
        # Set a flag to stop the run loop
        self.running = False
        # Wait for the thread to finish
        self.thread.join()

    def run(self):
        self.running = True
        
        # Calculate time until next UTC midnight
        def time_until_utc_midnight():
            now = datetime.datetime.now(pytz.utc)
            tomorrow = now.date() + datetime.timedelta(days=1)
            midnight = datetime.datetime.combine(tomorrow, datetime.time.min)
            return (midnight - now).total_seconds()
        schedule.every(time_until_utc_midnight()).seconds.do(self.reset_daily_cash)
        # For testing: run every minute instead of daily (if you think you're gonna be sneaky and reset this here, we check it on validator side`:) `)
        #schedule.every(1).minutes.do(self.reset_daily_cash)
        
        while self.running:
            schedule.run_pending()
            time.sleep(10)  # Check more frequently for testing

            # Reschedule for the next UTC midnight after each run
            if not schedule.jobs:
                schedule.every(time_until_utc_midnight()).seconds.do(self.reset_daily_cash)