import sqlite3
import os
import shutil
from datetime import datetime

# Define the correct database path
DB_DIR = "./bettensor/validator/state"
DB_NAME = "validator.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)
GOOD_BACKUP = os.path.join(DB_DIR, "validator_backup_20241204_085157.db")  # The known good backup

def cleanup_db_files(db_path):
    """Clean up database files including WAL and SHM."""
    files_to_remove = [
        db_path,
        f"{db_path}-wal",
        f"{db_path}-shm",
        f"{db_path}-journal"
    ]
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.unlink(file)
                print(f"Removed {file}")
            except Exception as e:
                print(f"Warning: Could not remove {file}: {e}")

def repair_backup():
    """Repair the backup database by copying it to a new file and reindexing."""
    print(f"Attempting to repair backup database...")
    temp_db = os.path.join(DB_DIR, "temp_repair.db")
    
    try:
        # Create a new database
        with sqlite3.connect(temp_db) as dest_conn:
            dest_conn.execute("PRAGMA foreign_keys = OFF")
            dest_conn.execute("PRAGMA journal_mode = DELETE")
            
            # Copy all tables from the backup
            with sqlite3.connect(GOOD_BACKUP) as src_conn:
                src_conn.execute("PRAGMA foreign_keys = OFF")
                src_conn.execute("PRAGMA journal_mode = DELETE")
                
                # Get all table names
                tables = src_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
                ).fetchall()
                
                successful_tables = []
                
                for (table_name,) in tables:
                    try:
                        print(f"Copying table {table_name}...")
                        # Get table schema
                        schema = src_conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()[0]
                        dest_conn.execute(schema)
                        
                        # Copy data in smaller chunks
                        offset = 0
                        chunk_size = 1000
                        while True:
                            try:
                                data = src_conn.execute(f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}").fetchall()
                                if not data:
                                    break
                                    
                                placeholders = ','.join(['?' for _ in data[0]])
                                dest_conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", data)
                                offset += chunk_size
                                
                            except Exception as e:
                                print(f"Warning: Error copying chunk from {table_name} at offset {offset}: {e}")
                                break
                                
                        successful_tables.append(table_name)
                        print(f"Successfully copied table {table_name}")
                        
                    except Exception as e:
                        print(f"Warning: Could not copy table {table_name}: {e}")
                        continue
                
                if not successful_tables:
                    raise Exception("No tables could be copied successfully")
                
                print(f"Successfully copied tables: {', '.join(successful_tables)}")
                
                # Copy sqlite_sequence data if it exists
                try:
                    seq_data = src_conn.execute("SELECT * FROM sqlite_sequence").fetchall()
                    if seq_data:
                        print("Copying sqlite_sequence data...")
                        for name, seq in seq_data:
                            if name in successful_tables:
                                dest_conn.execute("INSERT INTO sqlite_sequence VALUES (?, ?)", (name, seq))
                except Exception as e:
                    print(f"Warning: Could not copy sqlite_sequence data: {e}")
                
                # Recreate indices for successful tables
                indices = src_conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL AND tbl_name IN (" + 
                    ','.join('?' for _ in successful_tables) + ")",
                    successful_tables
                ).fetchall()
                
                for (idx_sql,) in indices:
                    try:
                        dest_conn.execute(idx_sql)
                    except Exception as e:
                        print(f"Warning: Could not recreate index: {e}")
                
            integrity_check = dest_conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity_check != "ok":
                print(f"Warning: Integrity check failed: {integrity_check}")
            
        # Replace the original backup with the repaired one
        if os.path.exists(temp_db):
            repaired_backup = os.path.join(DB_DIR, "validator_backup_repaired.db")
            shutil.copy2(temp_db, repaired_backup)
            cleanup_db_files(temp_db)
            return repaired_backup
            
    except Exception as e:
        print(f"Error repairing backup: {e}")
        if os.path.exists(temp_db):
            cleanup_db_files(temp_db)
        return None

def verify_database(conn):
    """Verify database integrity."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA integrity_check")
    integrity_result = cursor.fetchone()[0]
    cursor.execute("PRAGMA quick_check")
    quick_check = cursor.fetchone()[0]
    
    # Also verify that required tables exist
    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
    tables = [row[0] for row in cursor.execute(tables_query)]
    required_tables = {
        'miner_stats', 'game_data', 'predictions', 'scores', 'score_state',
        'entropy_game_pools', 'entropy_predictions', '_init_check', 'db_version',
        'entropy_closed_games', 'entropy_miner_scores', 'entropy_system_state',
        'keys', 'miner_stats_backup'
    }
    
    missing_tables = required_tables - set(tables)
    if missing_tables:
        print("Missing required tables:", missing_tables)
        print("Found tables:", tables)
        return False
    
    return integrity_result == "ok" and quick_check == "ok"

def repair_database():
    """Repair the database by cleaning up WAL/SHM and restoring from known good backup."""
    if not os.path.exists(GOOD_BACKUP):
        print(f"Error: Known good backup not found at {GOOD_BACKUP}")
        return False
    
    # First repair the backup
    repaired_backup = repair_backup()
    if not repaired_backup:
        print("Failed to repair backup database")
        return False
        
    # Create state directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)
    
    print("Cleaning up existing database files...")
    cleanup_db_files(DB_PATH)
    
    try:
        print(f"Restoring from repaired backup: {repaired_backup}")
        shutil.copy2(repaired_backup, DB_PATH)
        
        print("Verifying restored database...")
        with sqlite3.connect(DB_PATH) as conn:
            # Enable WAL mode and foreign keys
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            
            if verify_database(conn):
                print("Database restored and verified successfully!")
                return True
            else:
                print("Database verification failed after restore.")
                # If verification fails, don't leave a corrupt database
                cleanup_db_files(DB_PATH)
                return False
                
    except Exception as e:
        print(f"Error repairing database: {e}")
        return False

def main():
    print("Starting database repair process...")
    if repair_database():
        print("\nDatabase has been repaired successfully.")
        print("The database has been restored from the repaired backup.")
    else:
        print("\nDatabase repair failed.")
        print("The database could not be restored. Please check the backup files in the state directory.")

if __name__ == "__main__":
    main() 