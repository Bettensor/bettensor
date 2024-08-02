import sqlite3
import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
import os
import bittensor as bt
from packaging import version
import datetime
import time
from typing import List, Dict, Any
import json

def get_sqlite_connection(sqlite_path: str) -> sqlite3.Connection:
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")
    return sqlite3.connect(sqlite_path)

def get_postgres_connection() -> psycopg2.connection:
    try:
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'bettensor'),
            user=os.getenv('DB_USER', 'bettensor'),
            password=os.getenv('DB_PASSWORD', 'your_password_here')
        )
    except psycopg2.Error as e:
        bt.logging.error(f"Failed to connect to PostgreSQL: {e}")
        raise

def get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]

def sanitize_value(value: Any) -> Any:
    if isinstance(value, (int, float, bool, type(None))):
        return value
    return str(value)

def migrate_table(sqlite_conn: sqlite3.Connection, pg_conn: psycopg2.connection, table_name: str):
    sqlite_cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()

    # Get column names
    columns = get_table_columns(sqlite_cursor, table_name)

    # Fetch data in batches
    batch_size = 1000
    offset = 0
    total_migrated = 0

    while True:
        sqlite_cursor.execute(f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}")
        rows = sqlite_cursor.fetchall()
        
        if not rows:
            break

        # Prepare data for insertion
        sanitized_rows = [[sanitize_value(value) for value in row] for row in rows]

        # Construct the INSERT query
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns))
        )

        # Execute the INSERT query
        try:
            pg_cursor.executemany(insert_query, sanitized_rows)
            pg_conn.commit()
            total_migrated += len(rows)
            bt.logging.info(f"Migrated {len(rows)} rows from {table_name}")
        except psycopg2.Error as e:
            pg_conn.rollback()
            bt.logging.error(f"Error migrating data for {table_name}: {e}")
            bt.logging.error(f"Problematic data: {json.dumps(sanitized_rows, default=str)}")

        offset += batch_size

    bt.logging.info(f"Total migrated rows for {table_name}: {total_migrated}")

def validate_migration(sqlite_conn: sqlite3.Connection, pg_conn: psycopg2.connection, table_name: str):
    sqlite_cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()

    sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    sqlite_count = sqlite_cursor.fetchone()[0]

    pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    pg_count = pg_cursor.fetchone()[0]

    if sqlite_count != pg_count:
        bt.logging.warning(f"Migration validation failed for {table_name}. SQLite: {sqlite_count}, PostgreSQL: {pg_count}")
    else:
        bt.logging.info(f"Migration validation successful for {table_name}. {pg_count} rows migrated.")

def create_postgres_tables(pg_cursor: psycopg2.cursor):
    tables = [
        ("predictions", """
            CREATE TABLE IF NOT EXISTS predictions (
                predictionID TEXT PRIMARY KEY,
                teamGameID TEXT,
                minerID TEXT,
                predictionDate TIMESTAMP,
                predictedOutcome TEXT,
                teamA TEXT,
                teamB TEXT,
                wager DOUBLE PRECISION,
                teamAodds DOUBLE PRECISION,
                teamBodds DOUBLE PRECISION,
                tieOdds DOUBLE PRECISION,
                outcome TEXT
            )
        """),
        ("games", """
            CREATE TABLE IF NOT EXISTS games (
                gameID TEXT PRIMARY KEY,
                teamA TEXT,
                teamAodds DOUBLE PRECISION,
                teamB TEXT,
                teamBodds DOUBLE PRECISION,
                sport TEXT,
                league TEXT,
                externalID TEXT,
                createDate TIMESTAMP,
                lastUpdateDate TIMESTAMP,
                eventStartDate TIMESTAMP,
                active BOOLEAN,
                outcome TEXT,
                tieOdds DOUBLE PRECISION,
                canTie BOOLEAN
            )
        """),
        ("miner_stats", """
            CREATE TABLE IF NOT EXISTS miner_stats (
                miner_hotkey TEXT PRIMARY KEY,
                miner_uid INTEGER,
                miner_rank INTEGER,
                miner_cash DOUBLE PRECISION,
                miner_current_incentive DOUBLE PRECISION,
                miner_last_prediction_date TIMESTAMP,
                miner_lifetime_earnings DOUBLE PRECISION,
                miner_lifetime_wager DOUBLE PRECISION,
                miner_lifetime_predictions INTEGER,
                miner_lifetime_wins INTEGER,
                miner_lifetime_losses INTEGER,
                miner_win_loss_ratio DOUBLE PRECISION,
                last_daily_reset TIMESTAMP
            )
        """),
        ("database_version", """
            CREATE TABLE IF NOT EXISTS database_version (
                version TEXT PRIMARY KEY,
                timestamp TIMESTAMP
            )
        """)
    ]

    for table_name, create_query in tables:
        try:
            pg_cursor.execute(create_query)
            bt.logging.info(f"Created table: {table_name}")
        except psycopg2.Error as e:
            bt.logging.error(f"Error creating table {table_name}: {e}")

def migrate_to_postgres():
    # SQLite connection
    sqlite_path = os.path.expanduser("~/bettensor/data/miner.db")
    sqlite_conn = get_sqlite_connection(sqlite_path)

    # PostgreSQL connection
    pg_conn = get_postgres_connection()
    pg_conn.autocommit = False

    try:
        with pg_conn.cursor(cursor_factory=DictCursor) as pg_cursor:
            # Create tables in PostgreSQL if they don't exist
            create_postgres_tables(pg_cursor)

            # Migrate tables
            tables_to_migrate = ['predictions', 'games', 'miner_stats', 'database_version']
            for table in tables_to_migrate:
                migrate_table(sqlite_conn, pg_conn, table)
                validate_migration(sqlite_conn, pg_conn, table)

        # Update the database version
        with pg_conn.cursor() as pg_cursor:
            pg_cursor.execute("""
                INSERT INTO database_version (version, timestamp)
                VALUES (%s, CURRENT_TIMESTAMP)
                ON CONFLICT (version) DO UPDATE SET timestamp = CURRENT_TIMESTAMP
            """, (bt.__database_version__,))

        bt.logging.info("Migration to PostgreSQL completed successfully")
    except Exception as e:
        bt.logging.error(f"Error during migration to PostgreSQL: {e}")
        pg_conn.rollback()
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    migrate_to_postgres()