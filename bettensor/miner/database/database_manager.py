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
from psycopg2.pool import SimpleConnectionPool
import os
import bittensor as bt
from contextlib import contextmanager
from psycopg2.extras import DictCursor
import traceback

class DatabaseManager:
    def __init__(self, db_name, db_user, db_password, db_host=None, db_port=None, max_connections=10):
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.max_connections = max_connections
        self.connection_pool = None
        self.initialize_connection_pool()

    def initialize_connection_pool(self):
        try:
            connection_params = {
                "dbname": self.db_name,
                "user": self.db_user,
                "password": self.db_password,
            }
            if self.db_host:
                connection_params["host"] = self.db_host
            if self.db_port:
                connection_params["port"] = self.db_port

            self.connection_pool = SimpleConnectionPool(
                1, self.max_connections, **connection_params
            )
            bt.logging.info(f"PostgreSQL connection pool initialized with {self.max_connections} connections")
        except psycopg2.Error as e:
            bt.logging.error(f"Error initializing PostgreSQL connection pool: {e}")
            raise

    def execute_query(self, query, params=None):
        with self.get_cursor() as cursor:
            try:
                if params is not None:
                    if isinstance(params, dict):
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query, (params,) if not isinstance(params, (list, tuple)) else params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith("SELECT"):
                    result = cursor.fetchall()
                    return result
                else:
                    cursor.connection.commit()
                return None
            except Exception as e:
                cursor.connection.rollback()
                bt.logging.error(f"Error in execute_query: {str(e)}")
                bt.logging.error(f"Query: {query}")
                bt.logging.error(f"Params: {params}")
                raise

    def close(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            bt.logging.info("All database connections closed")

    def update_parameters(self, db_path=None, max_connections=None, db_host=None, db_name=None, db_user=None, db_password=None):
        if db_path is not None:
            self.db_path = db_path
        if max_connections is not None:
            self.max_connections = max_connections
        if db_host is not None:
            self.db_host = db_host
        if db_name is not None:
            self.db_name = db_name
        if db_user is not None:
            self.db_user = db_user
        if db_password is not None:
            self.db_password = db_password

    def initialize_sqlite_connection(self):
        try:
            sqlite_path = os.path.join(os.path.dirname(self.db_path), 'miner.db')
            self.connection_pool = sqlite3.connect(sqlite_path)
            bt.logging.info(f"SQLite database initialized at {sqlite_path}")
        except sqlite3.Error as e:
            bt.logging.error(f"Error initializing SQLite database: {e}")
            raise

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
            if isinstance(self.connection_pool, psycopg2.pool.SimpleConnectionPool):
                cursor.execute("""
                    INSERT INTO database_version (version, timestamp)
                    VALUES (%s, CURRENT_TIMESTAMP)
                """, (__database_version__,))
            else:
                cursor.execute("""
                    INSERT INTO database_version (version, timestamp)
                    VALUES (?, CURRENT_TIMESTAMP)
                """, (__database_version__,))
        else:
            bt.logging.info("Database is already at the latest version")

    @contextmanager
    def get_cursor(self):
        if isinstance(self.connection_pool, psycopg2.pool.SimpleConnectionPool):
            connection = self.connection_pool.getconn()
            try:
                with connection:
                    with connection.cursor(cursor_factory=DictCursor) as cursor:
                        yield cursor
            finally:
                self.connection_pool.putconn(connection)
        elif isinstance(self.connection_pool, sqlite3.Connection):
            cursor = self.connection_pool.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
        else:
            raise RuntimeError("No valid database connection available")

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

def get_db_manager(max_connections=10, state_manager=None, miner_uid=None, redis_interface=None):
    db_path = os.getenv('DB_PATH', os.path.expanduser("~/bettensor/data/miner.db"))
    db_host = os.getenv('DB_HOST', 'localhost')
    db_name = os.getenv('DB_NAME', 'bettensor')
    db_user = os.getenv('DB_USER', 'root')
    db_password = os.getenv('DB_PASSWORD', 'bettensor_password')
    return DatabaseManager.get_instance(db_path=db_path, max_connections=max_connections,
                                        db_host=db_host, db_name=db_name,
                                        db_user=db_user, db_password=db_password)