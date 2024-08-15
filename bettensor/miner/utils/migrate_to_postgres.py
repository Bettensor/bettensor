import os
import time
import sqlite3
import subprocess

try:
    import psycopg2
    from psycopg2.extras import DictCursor
    from psycopg2.extensions import cursor as psycopg2_cursor
except ImportError:
    print("psycopg2 not found. Installing...")
    subprocess.check_call(["pip", "install", "--no-cache-dir", "psycopg2-binary"])
    import psycopg2
    from psycopg2.extras import DictCursor
    from psycopg2.extensions import cursor as psycopg2_cursor



def get_postgres_connection():
    try:
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'bettensor'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'bettensor_password')
        )
    except psycopg2.Error as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        return None

def wait_for_postgres(max_retries=5, retry_delay=5):
    for attempt in range(max_retries):
        try:
            pg_conn = get_postgres_connection()
            pg_conn.close()
            print("Successfully connected to PostgreSQL.")
            return
        except psycopg2.OperationalError:
            if attempt < max_retries - 1:
                print(f"PostgreSQL not ready (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to PostgreSQL after multiple attempts.")
                raise

def create_postgres_tables(pg_cursor):
    tables = [
        ("predictions", """
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
                teamBodds REAL,
                tieOdds REAL,
                outcome TEXT
            )
        """),
        ("games", """
            CREATE TABLE IF NOT EXISTS games (
                gameID TEXT PRIMARY KEY,
                teamA TEXT,
                teamAodds REAL,
                teamB TEXT,
                teamBodds REAL,
                sport TEXT,
                league TEXT,
                externalID TEXT UNIQUE,
                createDate TEXT,
                lastUpdateDate TEXT,
                eventStartDate TEXT,
                active INTEGER,
                outcome TEXT,
                tieOdds REAL,
                canTie BOOLEAN
            )
        """),
        ("miner_stats", """
            CREATE TABLE IF NOT EXISTS miner_stats (
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
    ]

    for table_name, create_query in tables:
        try:
            pg_cursor.execute(create_query)
            print(f"Created table: {table_name}")
        except psycopg2.Error as e:
            print(f"Error creating table {table_name}: {e}")

def migrate_data(sqlite_conn, pg_cursor):
    tables = ['predictions', 'games', 'miner_stats']
    for table in tables:
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute(f"SELECT * FROM {table}")
        rows = sqlite_cursor.fetchall()
        columns = [description[0] for description in sqlite_cursor.description]
        
        if rows:
            placeholders = ','.join(['%s'] * len(columns))
            insert_query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
            
            for row in rows:
                # Convert canTie to boolean for the games table
                if table == 'games':
                    row = list(row)
                    can_tie_index = columns.index('canTie')
                    row[can_tie_index] = bool(row[can_tie_index])
                
                pg_cursor.execute(insert_query, row)

    print(f"Data migration completed for tables: {', '.join(tables)}")

def database_exists(cursor, db_name):
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    return cursor.fetchone() is not None

def create_database_if_not_exists():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'bettensor_password')
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        if not database_exists(cursor, os.getenv('DB_NAME', 'bettensor')):
            cursor.execute(f"CREATE DATABASE {os.getenv('DB_NAME', 'bettensor')}")
            print(f"Database {os.getenv('DB_NAME', 'bettensor')} created successfully")
        else:
            print(f"Database {os.getenv('DB_NAME', 'bettensor')} already exists")
        
        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        if "already exists" in str(e):
            print(f"Database {os.getenv('DB_NAME', 'bettensor')} already exists")
        else:
            raise

def setup_postgres(sqlite_db_path):
    # Backup the SQLite database
    backup_dir = os.path.join(os.path.dirname(sqlite_db_path), 'backups')
    try:
        from bettensor.miner.utils.database_backup import backup_database
        backup_file = backup_database(sqlite_db_path, backup_dir)
        if backup_file:
            print(f"SQLite database backed up to {backup_file}")
        else:
            print("Failed to backup SQLite database. Proceeding with caution.")
    except ImportError:
        print("Warning: Could not import backup function. Skipping backup.")
    except Exception as e:
        print(f"Error during backup: {e}. Proceeding with caution.")

    try:
        create_database_if_not_exists()
    except Exception as e:
        print(f"Error creating database: {e}. Proceeding with existing database.")

    try:
        wait_for_postgres()
        pg_conn = get_postgres_connection()

        if not pg_conn:
            print("Failed to connect to PostgreSQL. Please set up PostgreSQL manually.")
            return

        if not os.path.exists(sqlite_db_path):
            print(f"SQLite database file not found: {sqlite_db_path}")
            return

        sqlite_conn = sqlite3.connect(sqlite_db_path)
        with pg_conn.cursor(cursor_factory=DictCursor) as pg_cursor:
            create_postgres_tables(pg_cursor)
            migrate_data(sqlite_conn, pg_cursor)
        pg_conn.commit()
        print("PostgreSQL setup completed successfully")
    except Exception as e:
        print(f"Error during PostgreSQL setup: {e}")
        if 'pg_conn' in locals():
            pg_conn.rollback()
    finally:
        if 'pg_conn' in locals():
            pg_conn.close()
        if 'sqlite_conn' in locals():
            sqlite_conn.close()

    print("Migration process completed. Please check the logs for any errors.")
    print("If you encountered any issues, please set up PostgreSQL manually and rerun the migration.")

if __name__ == "__main__":
    sqlite_db_path = "./data/miner.db"
    setup_postgres(sqlite_db_path)