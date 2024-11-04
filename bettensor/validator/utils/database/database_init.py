"""
database_init.py

This file contains the code for initializing the database for the validator.

"""


def initialize_database():
    return [
        # Backup existing data
        """
        CREATE TABLE IF NOT EXISTS miner_stats_backup AS 
        SELECT * FROM miner_stats;
        """,
        
        # Drop existing table and indices
        "DROP INDEX IF EXISTS idx_miner_stats_hotkey",
        "DROP TABLE IF EXISTS miner_stats",
        
        # Create new table with correct schema
        """
        CREATE TABLE miner_stats (
            miner_uid INTEGER PRIMARY KEY,
            miner_hotkey TEXT UNIQUE,
            miner_coldkey TEXT,
            miner_rank INTEGER,
            miner_status TEXT,
            miner_cash REAL,
            miner_current_incentive REAL,
            miner_current_tier INTEGER,
            miner_current_scoring_window INTEGER,
            miner_current_composite_score REAL,
            miner_current_entropy_score REAL,
            miner_current_sharpe_ratio REAL,
            miner_current_sortino_ratio REAL,
            miner_current_roi REAL,
            miner_current_clv_avg REAL,
            miner_last_prediction_date TEXT,
            miner_lifetime_earnings REAL,
            miner_lifetime_wager_amount REAL,
            miner_lifetime_roi REAL,
            miner_lifetime_predictions INTEGER,
            miner_lifetime_wins INTEGER,
            miner_lifetime_losses INTEGER,
            miner_win_loss_ratio REAL
        )
        """,
        
        # Restore data from backup
        """
        INSERT INTO miner_stats 
        SELECT * FROM miner_stats_backup
        ON CONFLICT(miner_uid) DO UPDATE SET
            miner_hotkey = excluded.miner_hotkey,
            miner_coldkey = excluded.miner_coldkey,
            miner_rank = excluded.miner_rank,
            miner_status = excluded.miner_status,
            miner_cash = excluded.miner_cash,
            miner_current_incentive = excluded.miner_current_incentive,
            miner_current_tier = excluded.miner_current_tier,
            miner_current_scoring_window = excluded.miner_current_scoring_window,
            miner_current_composite_score = excluded.miner_current_composite_score,
            miner_current_entropy_score = excluded.miner_current_entropy_score,
            miner_current_sharpe_ratio = excluded.miner_current_sharpe_ratio,
            miner_current_sortino_ratio = excluded.miner_current_sortino_ratio,
            miner_current_roi = excluded.miner_current_roi,
            miner_current_clv_avg = excluded.miner_current_clv_avg,
            miner_last_prediction_date = excluded.miner_last_prediction_date,
            miner_lifetime_earnings = excluded.miner_lifetime_earnings,
            miner_lifetime_wager_amount = excluded.miner_lifetime_wager_amount,
            miner_lifetime_roi = excluded.miner_lifetime_roi,
            miner_lifetime_predictions = excluded.miner_lifetime_predictions,
            miner_lifetime_wins = excluded.miner_lifetime_wins,
            miner_lifetime_losses = excluded.miner_lifetime_losses,
            miner_win_loss_ratio = excluded.miner_win_loss_ratio
        """,
        
        # Drop backup table
        "DROP TABLE IF EXISTS miner_stats_backup",
        
        # Add index that allows NULL values in hotkey
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_miner_stats_hotkey 
        ON miner_stats(miner_hotkey) 
        WHERE miner_hotkey IS NOT NULL
        """
    ]
