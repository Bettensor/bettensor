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

import queue
from threading import Thread

class MinerStatsHandler:
    """
    This class is used to store miner stats and perform calculations on the data. It is instantiated for each validator and miner.

    """

    def __init__(self, miner):
        self.miner = miner
        self.queue = queue.Queue()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                method, args, kwargs = self.queue.get(timeout=1)
                if method == "stop":
                    break
                getattr(self, method)(*args, **kwargs)
            except queue.Empty:
                continue

    def stop(self):
        self.queue.put(("stop", [], {}))
        self.thread.join()

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            self.queue.put((name, args, kwargs))
        return wrapper

    # Update other methods to use self.miner.db and self.miner.cursor
    # For example:
    def update_miner_row(self, stats):
        with self.miner.db_lock:
            self.miner.cursor.execute(
                """UPDATE miner_stats SET
                miner_lifetime_wins = ?,
                miner_lifetime_losses = ?,
                miner_lifetime_ratio = ?,
                miner_lifetime_earnings = ?
                WHERE miner_hotkey = ?""",
                (
                    stats.miner_lifetime_wins,
                    stats.miner_lifetime_losses,
                    stats.miner_lifetime_ratio,
                    stats.miner_lifetime_earnings,
                    self.miner.hotkey
                )
            )
            self.miner.db.commit()

    def reset_daily_cash(self):
        """
        This method resets the daily cash of every miner to 1000, executed at 00:00 UTC
        """
        with self.miner.db_lock:
            self.miner.cursor.execute(
                """
            UPDATE miner_stats
            SET miner_cash = 1000
            """
            )
            self.miner.db.commit()
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
        bt.logging.info(f"init_miner_row() | Miner hotkey: {miner_hotkey}, Miner uid: {miner_uid}")

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

        with self.miner.db_lock:
            self.miner.cursor.execute(
                """
        SELECT * FROM miner_stats
        WHERE miner_hotkey = ?
        """,
                (miner_hotkey,),
            )
            if self.miner.cursor.fetchone():
                #update UID if necessary
                self.miner.cursor.execute(
                    """
                        UPDATE miner_stats
                        SET miner_uid = ?
                        WHERE miner_hotkey = ?
                        """,
                        (miner_uid, miner_hotkey),
                        )
                self.miner.db.commit()
                return True
            else:
                pass

            self.miner.cursor.execute(
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
            self.miner.db.commit()

        return True

    def return_miner_stats(self, miner_hotkey: str) -> MinerStats:
        """
        This method returns the miner row from the database as a MinerStats object, using the create method from the MinerStats class
        """
        with self.miner.db_lock:
            self.miner.cursor.execute(
                """
        SELECT * FROM miner_stats
        WHERE miner_hotkey = ?
        """,
                (miner_hotkey,),
            )
            row = self.miner.cursor.fetchone()

            return MinerStats.create(row)

    def run(self):
        self.running = True
        
        # Calculate time until next UTC midnight
        def time_until_utc_midnight():
            now = datetime.datetime.now(pytz.utc)
            tomorrow = now.date() + datetime.timedelta(days=1)
            midnight = datetime.datetime.combine(tomorrow, datetime.time.min, tzinfo=pytz.utc)
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
