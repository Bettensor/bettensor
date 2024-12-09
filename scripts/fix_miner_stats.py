import sqlite3
import os

DB_DIR = "./bettensor/validator/state"
DB_NAME = "validator.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)

def fix_null_scores():
    """Fix any null scores in miner_stats table."""
    print("Starting null score fix...")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check for null scores
            cursor.execute("""
                SELECT COUNT(*) FROM miner_stats 
                WHERE miner_current_composite_score IS NULL 
                   OR miner_current_sharpe_ratio IS NULL
                   OR miner_current_sortino_ratio IS NULL
                   OR miner_current_roi IS NULL
                   OR miner_current_clv_avg IS NULL
            """)
            null_count = cursor.fetchone()[0]
            print(f"Found {null_count} miners with null scores")
            
            if null_count == 0:
                print("No null scores found. No fixes needed.")
                return
            
            # Update null scores to 0
            cursor.execute("""
                UPDATE miner_stats 
                SET miner_current_composite_score = COALESCE(miner_current_composite_score, 0),
                    miner_current_sharpe_ratio = COALESCE(miner_current_sharpe_ratio, 0),
                    miner_current_sortino_ratio = COALESCE(miner_current_sortino_ratio, 0),
                    miner_current_roi = COALESCE(miner_current_roi, 0),
                    miner_current_clv_avg = COALESCE(miner_current_clv_avg, 0)
                WHERE miner_current_composite_score IS NULL 
                   OR miner_current_sharpe_ratio IS NULL
                   OR miner_current_sortino_ratio IS NULL
                   OR miner_current_roi IS NULL
                   OR miner_current_clv_avg IS NULL
            """)
            
            # Verify fix
            cursor.execute("""
                SELECT COUNT(*) FROM miner_stats 
                WHERE miner_current_composite_score IS NULL 
                   OR miner_current_sharpe_ratio IS NULL
                   OR miner_current_sortino_ratio IS NULL
                   OR miner_current_roi IS NULL
                   OR miner_current_clv_avg IS NULL
            """)
            remaining_nulls = cursor.fetchone()[0]
            print(f"Remaining null scores after fix: {remaining_nulls}")
            
            if remaining_nulls > 0:
                raise Exception(f"Failed to fix all null scores. {remaining_nulls} remain.")
            
            print("Successfully fixed all null scores!")
            
    except Exception as e:
        print(f"Error fixing null scores: {e}")
        raise

def cleanup_miner_stats():
    """Clean up miner_stats table to ensure it only has 256 rows."""
    print("Starting miner_stats cleanup...")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Get current count
            cursor.execute("SELECT COUNT(*) FROM miner_stats")
            current_count = cursor.fetchone()[0]
            print(f"Current number of rows in miner_stats: {current_count}")
            
            if current_count <= 256:
                print("Table already has 256 or fewer rows. No cleanup needed.")
                return
            
            # Create backup of current data
            print("Creating backup of miner_stats...")
            cursor.execute("DROP TABLE IF EXISTS miner_stats_backup_temp")
            cursor.execute("CREATE TABLE miner_stats_backup_temp AS SELECT * FROM miner_stats")
            
            # Get the most recent 256 rows based on last prediction date and highest earnings
            print("Identifying rows to keep...")
            cursor.execute("""
                WITH RankedMiners AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            ORDER BY 
                                CASE WHEN miner_last_prediction_date IS NULL THEN 0 ELSE 1 END DESC,
                                miner_last_prediction_date DESC,
                                miner_lifetime_earnings DESC,
                                miner_uid
                        ) as rn
                    FROM miner_stats
                )
                SELECT * FROM RankedMiners WHERE rn <= 256
            """)
            rows_to_keep = cursor.fetchall()
            
            if len(rows_to_keep) != 256:
                raise Exception(f"Expected 256 rows to keep, but got {len(rows_to_keep)}")
            
            # Drop and recreate the table
            print("Recreating miner_stats table...")
            cursor.execute("DROP TABLE miner_stats")
            cursor.execute("""
                CREATE TABLE miner_stats (
                    miner_uid INTEGER PRIMARY KEY,
                    miner_hotkey TEXT UNIQUE,
                    miner_coldkey TEXT,
                    miner_status TEXT,
                    miner_cash REAL DEFAULT 0,
                    miner_current_incentive REAL DEFAULT 0,
                    miner_current_tier INTEGER DEFAULT 1,
                    miner_current_scoring_window INTEGER DEFAULT 0,
                    miner_current_composite_score REAL DEFAULT 0,
                    miner_current_sharpe_ratio REAL DEFAULT 0,
                    miner_current_sortino_ratio REAL DEFAULT 0,
                    miner_current_roi REAL DEFAULT 0,
                    miner_current_clv_avg REAL DEFAULT 0,
                    miner_last_prediction_date TEXT,
                    miner_lifetime_earnings REAL DEFAULT 0,
                    miner_lifetime_wager_amount REAL DEFAULT 0,
                    miner_lifetime_roi REAL DEFAULT 0,
                    miner_lifetime_predictions INTEGER DEFAULT 0,
                    miner_lifetime_wins INTEGER DEFAULT 0,
                    miner_lifetime_losses INTEGER DEFAULT 0,
                    miner_win_loss_ratio REAL DEFAULT 0
                )
            """)
            
            # Reinsert the rows we want to keep, ensuring no nulls
            print("Reinserting rows with no null scores...")
            cursor.executemany("""
                INSERT INTO miner_stats (
                    miner_uid, miner_hotkey, miner_coldkey, miner_status,
                    miner_cash, miner_current_incentive, miner_current_tier,
                    miner_current_scoring_window, miner_current_composite_score,
                    miner_current_sharpe_ratio, miner_current_sortino_ratio,
                    miner_current_roi, miner_current_clv_avg,
                    miner_last_prediction_date, miner_lifetime_earnings,
                    miner_lifetime_wager_amount, miner_lifetime_roi,
                    miner_lifetime_predictions, miner_lifetime_wins,
                    miner_lifetime_losses, miner_win_loss_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE(?, 0), COALESCE(?, 0), COALESCE(?, 0), 
                    COALESCE(?, 0), COALESCE(?, 0), 
                    ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows_to_keep)
            
            # Verify the count
            cursor.execute("SELECT COUNT(*) FROM miner_stats")
            final_count = cursor.fetchone()[0]
            print(f"Final number of rows in miner_stats: {final_count}")
            
            if final_count != 256:
                raise Exception(f"Final count {final_count} does not match expected 256 rows")
            
            # Drop the temporary backup if everything succeeded
            cursor.execute("DROP TABLE miner_stats_backup_temp")
            print("Cleanup completed successfully!")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        # If we have a backup, restore it
        try:
            if conn:
                cursor.execute("SELECT COUNT(*) FROM miner_stats_backup_temp")
                if cursor.fetchone()[0] > 0:
                    print("Restoring from backup...")
                    cursor.execute("DROP TABLE miner_stats")
                    cursor.execute("ALTER TABLE miner_stats_backup_temp RENAME TO miner_stats")
                    print("Restored from backup.")
        except Exception as restore_error:
            print(f"Error during restore: {restore_error}")
        raise

if __name__ == "__main__":
    # First fix any null scores
    fix_null_scores()
    # Then do the regular cleanup
    cleanup_miner_stats() 