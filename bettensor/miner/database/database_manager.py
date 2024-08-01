import queue
import sqlite3
import os
import threading
import time
import bittensor as bt
from contextlib import asynccontextmanager, contextmanager
from queue import Queue
from bettensor.miner.utils.migrate import migrate_database
from bettensor import __database_version__
import asyncio
from concurrent.futures import ThreadPoolExecutor
from packaging import version

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path=None, max_connections=10):
        if db_path is None:
            # Use a relative path to ./data directory
            db_path = os.path.join(os.getcwd(), 'data', 'miner.db')
        self.db_path = db_path
        self.target_version = __database_version__
        bt.logging.info(f"Initializing DatabaseManager with target version: {self.target_version}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        

        self._initialize_database()
        self.connection_pool = Queue(maxsize=max_connections)
        for _ in range(max_connections):
            self.connection_pool.put(self._create_connection())

    def _create_connection(self):
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
                conn.execute("PRAGMA busy_timeout = 30000;")
                return conn
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    bt.logging.warning(f"Database is locked. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    raise
        raise sqlite3.OperationalError("Failed to create database connection after multiple attempts")

    def _initialize_database(self):
        bt.logging.info("Initializing database")
        conn = self._create_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Create database_version table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS database_version (
                        version TEXT PRIMARY KEY,
                        timestamp TEXT
                    )
                """)
                
                # Check current version
                cursor.execute("SELECT version FROM database_version ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                current_version = result[0] if result else '0.0.0'
                
                bt.logging.info(f"Current database version: {current_version}")
                
                if current_version != __database_version__:
                    bt.logging.info(f"Migrating database from {current_version} to {__database_version__}")
                    if not migrate_database(conn, self.db_path, __database_version__):
                        raise Exception("Database migration failed")
                    
                    bt.logging.info("Updating database_version table")
                    cursor.execute("""
                        INSERT OR REPLACE INTO database_version (version, timestamp)
                        VALUES (?, datetime('now'))
                    """, (__database_version__,))
                    conn.commit()
                    bt.logging.info(f"Database version updated to {__database_version__}")
                else:
                    bt.logging.info("Database is already at the latest version")
                
                # Ensure all necessary tables exist
                self._create_tables(cursor)
                
        except Exception as e:
            bt.logging.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()

    def _create_tables(self, cursor):
        bt.logging.info("Creating predictions table")
        cursor.execute("""
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
        """)
        
        bt.logging.info("Creating games table")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
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
        
        bt.logging.info("Creating miner_stats table")
        cursor.execute("""
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

    def get_table_info(self, table_name):
        with self.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table_name})")
            return cursor.fetchall()

    @classmethod
    def get_instance(cls, db_path, max_connections=10):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path, max_connections)
        return cls._instance

    @contextmanager
    def get_cursor(self):
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                connection = self.connection_pool.get(timeout=30)
                try:
                    cursor = connection.cursor()
                    yield cursor
                    connection.commit()
                    bt.logging.debug("Database transaction committed successfully")
                    return
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        bt.logging.warning(f"Database is locked. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        bt.logging.error(f"Database operation failed: {e}")
                        raise
                finally:
                    self.connection_pool.put(connection)
            except queue.Empty:
                if attempt < max_retries - 1:
                    bt.logging.warning(f"Connection pool is empty. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    bt.logging.error("Failed to get a database connection from the pool after multiple attempts")
                    raise sqlite3.OperationalError("Failed to get a database connection from the pool after multiple attempts")
        bt.logging.error("Failed to execute database operation after multiple attempts")
        raise sqlite3.OperationalError("Failed to execute database operation after multiple attempts")

    def close_all(self):
        while not self.connection_pool.empty():
            conn = self.connection_pool.get()
            conn.close()

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