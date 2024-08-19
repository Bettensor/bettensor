import os
import sys
import sqlite3
import psycopg2
import shutil
import subprocess
from datetime import datetime
import time
import bittensor as bt

def backup_database(db_path, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if db_path.endswith('.db'):
        # SQLite backup
        backup_file = os.path.join(backup_dir, f"bettensor_backup_{timestamp}.db")
        shutil.copy2(db_path, backup_file)
        bt.logging.info(f"SQLite database backed up to {backup_file}")
    else:
        # PostgreSQL backup
        backup_file = os.path.join(backup_dir, f"bettensor_backup_{timestamp}.sql")
        try:
            subprocess.run(
                f"pg_dump -h {os.getenv('DB_HOST')} -U {os.getenv('DB_USER')} -d {os.getenv('DB_NAME')} > {backup_file}",
                shell=True,
                check=True
            )
            bt.logging.info(f"PostgreSQL database backed up to {backup_file}")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Failed to backup PostgreSQL database: {e}")
            return None

    return backup_file

def get_postgres_connection():
    try:
        return psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT')
        )
    except psycopg2.Error as e:
        bt.logging.error(f"Failed to connect to PostgreSQL: {e}")
        return None

def wait_for_postgres(max_retries=5, retry_delay=5):
    for attempt in range(max_retries):
        try:
            pg_conn = get_postgres_connection()
            if pg_conn:
                pg_conn.close()
                bt.logging.info("Successfully connected to PostgreSQL.")
                return True
        except psycopg2.OperationalError:
            if attempt < max_retries - 1:
                bt.logging.info(f"PostgreSQL not ready (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                bt.logging.error("Failed to connect to PostgreSQL after multiple attempts.")
                return False

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
        """),
        ("model_params", """
            CREATE TABLE IF NOT EXISTS model_params (
                id SERIAL PRIMARY KEY,
                model_on BOOLEAN,
                wager_distribution_steepness INTEGER,
                fuzzy_match_percentage INTEGER,
                minimum_wager_amount FLOAT,
                max_wager_amount FLOAT,
                top_n_games INTEGER
            )
        """),
        ("miner_active", """
            CREATE TABLE IF NOT EXISTS miner_active (
                miner_uid TEXT PRIMARY KEY,
                last_active_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    ]

    for table_name, create_query in tables:
        try:
            pg_cursor.execute(create_query)
            bt.logging.info(f"Created table: {table_name}")
        except psycopg2.Error as e:
            bt.logging.error(f"Error creating table {table_name}: {e}")

def migrate_data(sqlite_conn, pg_cursor):
    tables = ['predictions', 'games', 'miner_stats']
    for table in tables:
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute(f"SELECT * FROM {table}")
        rows = sqlite_cursor.fetchall()
        columns = [description[0] for description in sqlite_cursor.description]
        
        if rows:
            placeholders = ','.join(['%s'] * len(columns))
            insert_query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
            
            for row in rows:
                # Convert canTie to boolean for the games table
                if table == 'games':
                    row = list(row)
                    can_tie_index = columns.index('canTie')
                    row[can_tie_index] = bool(row[can_tie_index])
                
                pg_cursor.execute(insert_query, row)

    bt.logging.info(f"Data migration completed for tables: {', '.join(tables)}")

def create_database_if_not_exists():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{os.getenv('DB_NAME')}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {os.getenv('DB_NAME')}")
            bt.logging.info(f"Database {os.getenv('DB_NAME')} created successfully")
        else:
            bt.logging.info(f"Database {os.getenv('DB_NAME')} already exists")
        
        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        bt.logging.error(f"Error creating database: {e}")
        raise

def migrate_sqlite(sqlite_conn, sqlite_cursor):
    # Add SQLite migration logic here if needed
    pass

def setup_postgres(sqlite_db_path):
    # Backup the SQLite database
    backup_file = backup_database(sqlite_db_path, os.getenv('BACKUP_DIR'))
    if backup_file:
        bt.logging.info(f"SQLite database backed up to {backup_file}")
    else:
        bt.logging.error("Failed to backup SQLite database. Proceeding with caution.")

    try:
        create_database_if_not_exists()
    except Exception as e:
        bt.logging.error(f"Failed to create database: {e}")
        return

    if not wait_for_postgres():
        bt.logging.error("Failed to connect to PostgreSQL. Please set up PostgreSQL manually.")
        return

    try:
        pg_conn = get_postgres_connection()
        if not pg_conn:
            bt.logging.error("Failed to connect to PostgreSQL. Please set up PostgreSQL manually.")
            return

        sqlite_conn = None
        if os.path.exists(sqlite_db_path):
            sqlite_conn = sqlite3.connect(sqlite_db_path)
            sqlite_cursor = sqlite_conn.cursor()
            migrate_sqlite(sqlite_conn, sqlite_cursor)

        with pg_conn.cursor() as pg_cursor:
            create_postgres_tables(pg_cursor)
            if sqlite_conn:
                migrate_data(sqlite_conn, pg_cursor)
        pg_conn.commit()
        bt.logging.info("PostgreSQL setup completed successfully")
    except Exception as e:
        bt.logging.error(f"Error during PostgreSQL setup: {e}")
        if 'pg_conn' in locals():
            pg_conn.rollback()
    finally:
        if 'pg_conn' in locals():
            pg_conn.close()
        if 'sqlite_conn' in locals():
            sqlite_conn.close()

if __name__ == "__main__":
    sqlite_db_path = os.getenv('SQLITE_DB_PATH')
    setup_postgres(sqlite_db_path)