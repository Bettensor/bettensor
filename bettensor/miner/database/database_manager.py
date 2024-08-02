import queue
import sqlite3
import os
import threading
import time
import bittensor as bt
from contextlib import asynccontextmanager, contextmanager
from queue import Queue
from bettensor.miner.utils.migrate_sqlite import migrate_database
from bettensor import __database_version__
import asyncio
from concurrent.futures import ThreadPoolExecutor
from packaging import version

import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor
import os
import bittensor as bt
from contextlib import contextmanager

class DatabaseManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_connections=10, state_manager=None, miner_uid=None):
        if not hasattr(self, 'initialized'):
            self.max_connections = max_connections
            self.state_manager = state_manager
            self.miner_uid = miner_uid
            self.connection_pool = None
            self.initialize_connection_pool()
            self.initialize_database()
            self.initialized = True

    def initialize_connection_pool(self):
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, self.max_connections,
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'bettensor'),
                user=os.getenv('DB_USER', 'bettensor'),
                password=os.getenv('DB_PASSWORD', 'your_password_here')
            )
            bt.logging.info("PostgreSQL connection pool created successfully")
        except (Exception, psycopg2.Error) as error:
            bt.logging.error(f"Error while connecting to PostgreSQL: {error}")

    def initialize_database(self):
        bt.logging.info("Initializing database")
        with self.get_cursor() as cursor:
            self._create_tables(cursor)
            self._check_and_update_version(cursor)

    def _create_tables(self, cursor):
        bt.logging.info("Creating tables if they don't exist")
        
        # Create predictions table
        cursor.execute("""
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
        """)
        
        # Create games table
        cursor.execute("""
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
        """)
        
        # Create miner_stats table
        cursor.execute("""
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
        """)

        # Create database_version table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS database_version (
                version TEXT PRIMARY KEY,
                timestamp TIMESTAMP
            )
        """)

    def _check_and_update_version(self, cursor):
        from bettensor import __database_version__
        
        cursor.execute("SELECT version FROM database_version ORDER BY timestamp DESC LIMIT 1")
        result = cursor.fetchone()
        current_version = result[0] if result else '0.0.0'
        
        bt.logging.info(f"Current database version: {current_version}")
        
        if current_version != __database_version__:
            bt.logging.info(f"Updating database version to {__database_version__}")
            cursor.execute("""
                INSERT INTO database_version (version, timestamp)
                VALUES (%s, CURRENT_TIMESTAMP)
            """, (__database_version__,))
        else:
            bt.logging.info("Database is already at the latest version")

    @contextmanager
    def get_cursor(self):
        connection = self.connection_pool.getconn()
        try:
            with connection:
                with connection.cursor(cursor_factory=DictCursor) as cursor:
                    yield cursor
        finally:
            self.connection_pool.putconn(connection)

    def check_and_migrate(self):
        from bettensor.miner.utils.migrate_to_postgres import migrate_to_postgres
        import sqlite3
        from bettensor.miner.utils.migrate_sqlite import migrate_database
        from bettensor import __database_version__
        
        sqlite_path = os.path.expanduser("~/bettensor/data/miner.db")
        if os.path.exists(sqlite_path):
            bt.logging.info("SQLite database found. Checking for migration need.")
            
            # First, migrate SQLite database to the latest version
            try:
                with sqlite3.connect(sqlite_path) as sqlite_conn:
                    bt.logging.info("Migrating SQLite database to the latest version")
                    migrate_database(sqlite_conn, sqlite_path, __database_version__)
            except sqlite3.Error as e:
                bt.logging.error(f"Failed to migrate SQLite database: {e}")
                raise
            
            # Now, check if we need to migrate to PostgreSQL
            try:
                with self.get_cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM database_version")
                    if cursor.fetchone()[0] == 0:
                        bt.logging.info("PostgreSQL database is empty. Starting migration from SQLite.")
                        migrate_to_postgres()
                    else:
                        bt.logging.info("PostgreSQL database already contains data. Skipping migration.")
            except psycopg2.Error:
                bt.logging.error("Failed to connect to PostgreSQL. Please ensure the database is running.")
                raise
        else:
            bt.logging.info("No SQLite database found. Skipping migration.")
        
        # Finally, ensure PostgreSQL is at the latest version
        self.ensure_postgres_latest_version()

    def ensure_postgres_latest_version(self):
        with self.get_cursor() as cursor:
            cursor.execute("SELECT version FROM database_version ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
            current_version = result[0] if result else '0.0.0'
            
            if current_version != __database_version__:
                bt.logging.info(f"Updating PostgreSQL database version to {__database_version__}")
                cursor.execute("""
                    INSERT INTO database_version (version, timestamp)
                    VALUES (%s, CURRENT_TIMESTAMP)
                """, (__database_version__,))
            else:
                bt.logging.info("PostgreSQL database is already at the latest version")

def get_db_manager(max_connections=10, state_manager=None, miner_uid=None):
    return DatabaseManager(max_connections, state_manager, miner_uid)

def get_db_manager(max_connections=10, state_manager=None, miner_uid=None):
    config = bt.config()
    db_path = getattr(config, 'db_path', None)
    
    if db_path is None:
        bt.logging.warning("db_path not found in config. Using default path.")
        db_path = os.path.join(os.getcwd(), 'data', 'miner.db')
    
    bt.logging.info(f"Using database path: {db_path}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    return DatabaseManager.get_instance(db_path, max_connections)