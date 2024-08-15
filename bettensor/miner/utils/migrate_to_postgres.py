try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:
    print("psycopg2 not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "--no-cache-dir", "psycopg2-binary"])
    import psycopg2
    from psycopg2.extras import DictCursor
import os
import bittensor as bt
import time
import sqlite3
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
        bt.logging.error(f"Failed to connect to PostgreSQL: {e}")
        raise

def wait_for_postgres(max_retries=5, retry_delay=5):
    for attempt in range(max_retries):
        try:
            pg_conn = get_postgres_connection()
            pg_conn.close()
            bt.logging.info("Successfully connected to PostgreSQL.")
            return
        except psycopg2.OperationalError:
            if attempt < max_retries - 1:
                bt.logging.warning(f"PostgreSQL not ready (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                bt.logging.error("Failed to connect to PostgreSQL after multiple attempts.")
                raise

def create_postgres_tables(pg_cursor):
    tables = [
        ("games", """
            CREATE TABLE IF NOT EXISTS games (
                gameID UUID PRIMARY KEY,
                teamA TEXT NOT NULL,
                teamAodds REAL,
                teamB TEXT NOT NULL,
                teamBodds REAL,
                sport TEXT NOT NULL,
                league TEXT NOT NULL,
                externalID TEXT UNIQUE NOT NULL,
                createDate TIMESTAMP WITH TIME ZONE NOT NULL,
                lastUpdateDate TIMESTAMP WITH TIME ZONE NOT NULL,
                eventStartDate TIMESTAMP WITH TIME ZONE NOT NULL,
                active BOOLEAN NOT NULL,
                outcome TEXT,
                tieOdds REAL,
                canTie BOOLEAN
            )
        """),
        ("predictions", """
            CREATE TABLE IF NOT EXISTS predictions (
                predictionID UUID PRIMARY KEY,
                gameID UUID NOT NULL,
                minerID TEXT NOT NULL,
                prediction TEXT NOT NULL,
                wager REAL NOT NULL,
                odds REAL NOT NULL,
                createDate TIMESTAMP WITH TIME ZONE NOT NULL,
                lastUpdateDate TIMESTAMP WITH TIME ZONE NOT NULL,
                outcome TEXT,
                payout REAL,
                FOREIGN KEY (gameID) REFERENCES games(gameID)
            )
        """),
        ("miner_stats", """
            CREATE TABLE IF NOT EXISTS miner_stats (
                miner_hotkey TEXT PRIMARY KEY,
                miner_coldkey TEXT,
                miner_uid INTEGER,
                miner_rank INTEGER,
                miner_cash REAL,
                miner_current_incentive REAL,
                miner_last_prediction_date TIMESTAMP WITH TIME ZONE,
                miner_lifetime_earnings REAL,
                miner_lifetime_wager REAL,
                miner_lifetime_predictions INTEGER,
                miner_lifetime_wins INTEGER,
                miner_lifetime_losses INTEGER,
                miner_win_loss_ratio REAL,
                miner_status TEXT,
                last_daily_reset TIMESTAMP WITH TIME ZONE
            )
        """)
    ]

    for table_name, create_query in tables:
        try:
            pg_cursor.execute(create_query)
            bt.logging.info(f"Created table: {table_name}")
        except psycopg2.Error as e:
            bt.logging.error(f"Error creating table {table_name}: {e}")

def migrate_data(source_conn, dest_conn):
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()

    # Migrate games table
    source_cursor.execute("SELECT * FROM games")
    games = source_cursor.fetchall()
    for game in games:
        dest_cursor.execute("""
            INSERT INTO games (gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID,
                               createDate, lastUpdateDate, eventStartDate, active, outcome, tieOdds, canTie)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (externalID) DO NOTHING
        """, game)

    # Migrate predictions table
    source_cursor.execute("SELECT * FROM predictions")
    predictions = source_cursor.fetchall()
    for prediction in predictions:
        dest_cursor.execute("""
            INSERT INTO predictions (predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
                                     teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (predictionID) DO NOTHING
        """, prediction)

    # Migrate miner_stats table
    source_cursor.execute("SELECT * FROM miner_stats")
    miner_stats = source_cursor.fetchall()
    for stat in miner_stats:
        dest_cursor.execute("""
            INSERT INTO miner_stats (miner_hotkey, miner_uid, miner_cash, miner_current_incentive,
                                     miner_last_prediction_date, miner_lifetime_earnings, miner_lifetime_wager,
                                     miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses,
                                     miner_win_loss_ratio, last_daily_reset)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (miner_hotkey) DO NOTHING
        """, stat)

    dest_conn.commit()

def setup_postgres(sqlite_db_path):
    wait_for_postgres()
    pg_conn = get_postgres_connection()

    if not os.path.exists(sqlite_db_path):
        bt.logging.error(f"SQLite database file not found: {sqlite_db_path}")
        return

    try:
        sqlite_conn = sqlite3.connect(sqlite_db_path)
        with pg_conn.cursor(cursor_factory=DictCursor) as pg_cursor:
            create_postgres_tables(pg_cursor)
            migrate_data(sqlite_conn, pg_conn)
        pg_conn.commit()
        bt.logging.info("PostgreSQL setup completed successfully")
    except Exception as e:
        bt.logging.error(f"Error during PostgreSQL setup: {e}")
        pg_conn.rollback()
    finally:
        pg_conn.close()
        if 'sqlite_conn' in locals():
            sqlite_conn.close()

if __name__ == "__main__":
    sqlite_db_path = "./data/miner.db"
    setup_postgres(sqlite_db_path)