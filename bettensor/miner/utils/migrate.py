import sqlite3
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

def execute_with_retry(cursor, sql, params=None, max_retries=5, retry_delay=1):
    for attempt in range(max_retries):
        try:
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

def migrate_database(conn, db_path, target_version, max_retries=5, retry_delay=1):
    bt.logging.info(f"Starting database migration to version {target_version}")
    
    lock_file = os.path.join(os.path.dirname(db_path), 'db_migration.lock')
    db_lock = DatabaseLock(lock_file)
    
    for migration_attempt in range(max_retries):
        if not db_lock.acquire(timeout=60):
            bt.logging.error(f"Could not acquire database lock for migration (Attempt {migration_attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
            continue

        try:
            cursor = conn.cursor()
            
            # Get current version
            cursor.execute("SELECT version FROM database_version ORDER BY timestamp DESC, version DESC LIMIT 1")
            result = cursor.fetchone()
            current_version = result[0] if result else '0.0.0'
            
            bt.logging.info(f"Current database version: {current_version}")
            
            # Perform migrations based on versions
            while version.parse(current_version) < version.parse(target_version):
                bt.logging.info(f"Migrating from {current_version} to next version")
                if version.parse(current_version) < version.parse('0.0.1'):
                    # Migration logic for 0.0.0 to 0.0.1
                    execute_with_retry(cursor, """
                        CREATE TABLE IF NOT EXISTS predictions (
                            predictionID TEXT PRIMARY KEY,
                            teamGameID TEXT,
                            minerID TEXT,
                            predictionDate TEXT,
                            predictedOutcome TEXT,
                            teamA TEXT,
                            teamB TEXT,
                            wager REAL,
                            teamAodds REAL,
                            teamBodds REAL
                        )
                    """)
                    current_version = '0.0.1'
                elif version.parse(current_version) < version.parse('0.0.2'):
                    # Migration logic for 0.0.1 to 0.0.2
                    if not column_exists(cursor, 'predictions', 'outcome'):
                        execute_with_retry(cursor, "ALTER TABLE predictions ADD COLUMN outcome TEXT")
                    current_version = '0.0.2'
                elif version.parse(current_version) < version.parse('0.0.3'):
                    # Migration logic for 0.0.2 to 0.0.3
                    if not column_exists(cursor, 'predictions', 'tieOdds'):
                        execute_with_retry(cursor, "ALTER TABLE predictions ADD COLUMN tieOdds REAL")
                    current_version = '0.0.3'
                elif version.parse(current_version) < version.parse('0.0.4'):
                    # Migration logic for 0.0.3 to 0.0.4
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
                        INSERT OR REPLACE INTO predictions_new
                        SELECT predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
                               teamA, teamB, wager, teamAodds, teamBodds, 
                               COALESCE(tieOdds, 0) as tieOdds, outcome
                        FROM predictions
                    """)
                    
                    execute_with_retry(cursor, 'DROP TABLE IF EXISTS predictions')
                    execute_with_retry(cursor, 'ALTER TABLE predictions_new RENAME TO predictions')
                    
                    current_version = '0.0.4'
                elif version.parse(current_version) < version.parse('0.0.5'):
                    # Migration logic for 0.0.4 to 0.0.5
                    bt.logging.info("Migrating from 0.0.4 to 0.0.5: Updating miner_stats table")
                    
                    # Create a new table with the updated structure
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
                    
                    # Copy data from the old table to the new table
                    execute_with_retry(cursor, """
                        INSERT OR REPLACE INTO miner_stats_new (
                            miner_hotkey, miner_uid, miner_rank, miner_cash,
                            miner_current_incentive, miner_last_prediction_date, miner_lifetime_earnings,
                            miner_lifetime_wager, miner_lifetime_predictions, miner_lifetime_wins,
                            miner_lifetime_losses, miner_win_loss_ratio, last_daily_reset
                        )
                        SELECT
                            miner_hotkey,
                            miner_uid,
                            COALESCE(miner_rank, 0),
                            miner_cash,
                            COALESCE(miner_current_incentive, 0),
                            miner_last_prediction_date,
                            COALESCE(miner_lifetime_earnings, 0),
                            COALESCE(miner_lifetime_wager, 0),
                            COALESCE(miner_lifetime_predictions, 0),
                            COALESCE(miner_lifetime_wins, 0),
                            COALESCE(miner_lifetime_losses, 0),
                            COALESCE(miner_win_loss_ratio, 0),
                            COALESCE(last_daily_reset, datetime('now'))
                        FROM miner_stats
                    """)
                    
                    # Drop the old table and rename the new one
                    execute_with_retry(cursor, 'DROP TABLE IF EXISTS miner_stats')
                    execute_with_retry(cursor, 'ALTER TABLE miner_stats_new RENAME TO miner_stats')
                    
                    current_version = '0.0.5'
                else:
                    bt.logging.error(f"Unknown version {current_version}")
                    return False
                
            # Update the database version after all migration steps
            execute_with_retry(cursor, """
                INSERT OR REPLACE INTO database_version (version, timestamp)
                VALUES (?, datetime('now'))
            """, (target_version,))
            
            conn.commit()
            bt.logging.info(f"Database migration to version {target_version} completed successfully")
            return True
        
        except sqlite3.OperationalError as e:
            bt.logging.error(f"SQLite operational error during migration: {e}")
            conn.rollback()
            time.sleep(retry_delay)
        except Exception as e:
            bt.logging.error(f"Failed to migrate database: {e}")
            conn.rollback()
            return False
        finally:
            db_lock.release()
    
    bt.logging.error(f"Failed to migrate database after {max_retries} attempts")
    return False

if __name__ == "__main__":
    import os
    from bettensor import __database_version__
    db_path = os.environ.get('BETTENSOR_DB_PATH', os.path.expanduser('~/bettensor/data/miner.db'))
    with sqlite3.connect(db_path) as conn:
        migrate_database(conn, __database_version__)