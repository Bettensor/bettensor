import sqlite3
import traceback
import bittensor as bt
import time
import os
import signal
from contextlib import contextmanager
from bettensor.miner.utils.db_lock import DatabaseLock
from packaging import version

def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())

def execute_with_retry(cursor, sql, params=None, max_retries=5, retry_delay=1, timeout=30):
    for attempt in range(max_retries):
        try:
            cursor.execute(f"PRAGMA busy_timeout = {timeout * 1000}")  # Set timeout in milliseconds
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                bt.logging.warning(f"Database is locked. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise
        except sqlite3.Error as e:
            if attempt < max_retries - 1:
                bt.logging.warning(f"SQLite error occurred. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise

def table_exists(cursor, table_name):
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    return cursor.fetchone() is not None

def migrate_database(conn, db_path, target_version, max_retries=5, retry_delay=1):
    bt.logging.info(f"Starting database migration to version {target_version}")
    
    cursor = conn.cursor()
    
    # Get current version
    cursor.execute("SELECT version FROM database_version ORDER BY timestamp DESC, version DESC LIMIT 1")
    result = cursor.fetchone()
    current_version = result[0] if result else '0.0.0'
    
    bt.logging.info(f"Current database version: {current_version}")
    
    # Perform migrations based on versions
    while version.parse(current_version) < version.parse(target_version):
        bt.logging.info(f"Migrating from {current_version} to next version")
        if version.parse(current_version) < version.parse('0.0.5'):
            bt.logging.info("Starting migration to 0.0.5")
            
            # Migrate predictions table
            if table_exists(cursor, 'predictions'):
                bt.logging.info("Migrating predictions table")
                execute_with_retry(cursor, """
                    CREATE TABLE IF NOT EXISTS predictions_new (
                        predictionID TEXT PRIMARY KEY, 
                        teamGameID TEXT, 
                        minerID TEXT, 
                        predictionDate TEXT, 
                        predictedOutcome TEXT,
                        teamA TEXT,
                        teamB TEXT,
                        wager REAL,
                        teamAodds REAL,
                        teamBodds REAL,
                        tieOdds REAL,
                        outcome TEXT
                    )
                """)
                execute_with_retry(cursor, """
                    INSERT INTO predictions_new 
                    SELECT predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
                           teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
                    FROM predictions
                """)
                execute_with_retry(cursor, "DROP TABLE predictions")
                execute_with_retry(cursor, "ALTER TABLE predictions_new RENAME TO predictions")
            
            # Migrate games table
            if table_exists(cursor, 'games'):
                bt.logging.info("Migrating games table")
                execute_with_retry(cursor, """
                    CREATE TABLE IF NOT EXISTS games_new (
                        gameID TEXT PRIMARY KEY,
                        teamA TEXT,
                        teamAodds REAL,
                        teamB TEXT,
                        teamBodds REAL,
                        sport TEXT,
                        league TEXT,
                        externalID TEXT,
                        createDate TEXT,
                        lastUpdateDate TEXT,
                        eventStartDate TEXT,
                        active INTEGER,
                        outcome TEXT,
                        tieOdds REAL,
                        canTie BOOLEAN
                    )
                """)
                execute_with_retry(cursor, """
                    INSERT INTO games_new 
                    SELECT gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID,
                           createDate, lastUpdateDate, eventStartDate, active, outcome, tieOdds, canTie
                    FROM games
                """)
                execute_with_retry(cursor, "DROP TABLE games")
                execute_with_retry(cursor, "ALTER TABLE games_new RENAME TO games")
            
            # Migrate miner_stats table
            if table_exists(cursor, 'miner_stats'):
                bt.logging.info("Migrating miner_stats table")
                execute_with_retry(cursor, """
                    CREATE TABLE IF NOT EXISTS miner_stats_new (
                        miner_hotkey TEXT PRIMARY KEY,
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
                        last_daily_reset TEXT
                    )
                """)
                execute_with_retry(cursor, """
                    INSERT INTO miner_stats_new (
                        miner_hotkey, miner_uid, miner_rank, miner_cash, miner_current_incentive,
                        miner_last_prediction_date, miner_lifetime_earnings, miner_lifetime_wager,
                        miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses,
                        miner_win_loss_ratio
                    )
                    SELECT miner_hotkey, miner_uid, miner_rank, miner_cash, miner_current_incentive,
                           miner_last_prediction_date, miner_lifetime_earnings, miner_lifetime_wager,
                           miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses,
                           miner_win_loss_ratio
                    FROM miner_stats
                """)
                execute_with_retry(cursor, "UPDATE miner_stats_new SET last_daily_reset = datetime('now')")
                execute_with_retry(cursor, "DROP TABLE miner_stats")
                execute_with_retry(cursor, "ALTER TABLE miner_stats_new RENAME TO miner_stats")
            
            bt.logging.info("Migration to 0.0.5 completed")
            current_version = '0.0.5'
        else:
            bt.logging.error(f"Unknown version {current_version}")
            return False
        
    # Update the database version after all migration steps
    bt.logging.info(f"Updating database version to {target_version}")
    execute_with_retry(cursor, """
        INSERT OR REPLACE INTO database_version (version, timestamp)
        VALUES (?, datetime('now'))
    """, (target_version,))
    
    conn.commit()
    bt.logging.info(f"Database migration to version {target_version} completed successfully")
    return True

if __name__ == "__main__":
    import os
    from bettensor import __database_version__
    db_path = os.environ.get('BETTENSOR_DB_PATH', os.path.expanduser('~/bettensor/data/miner.db'))
    with sqlite3.connect(db_path) as conn:
        migrate_database(conn, db_path, __database_version__)