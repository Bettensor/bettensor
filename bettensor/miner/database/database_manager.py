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
            # Use a relative path within the bettensor directory
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'miner.db')
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
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def _initialize_database(self):
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("PRAGMA busy_timeout = 30000;")
                    cursor = conn.cursor()
                    
                    bt.logging.info("Checking for database_version table")
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='database_version'")
                    if cursor.fetchone() is None:
                        bt.logging.info("database_version table not found, creating it")
                        cursor.execute("""
                            CREATE TABLE database_version (
                                version TEXT PRIMARY KEY,
                                timestamp TEXT DEFAULT (datetime('now'))
                            )
                        """)
                        cursor.execute("INSERT INTO database_version (version) VALUES (?)", (self.target_version,))
                        conn.commit()
                        current_version = self.target_version
                        bt.logging.info(f"Initialized database_version table with version: {current_version}")
                    else:
                        bt.logging.info("database_version table found, checking for timestamp column")
                        cursor.execute("PRAGMA table_info(database_version)")
                        columns = [column[1] for column in cursor.fetchall()]
                        if 'timestamp' not in columns:
                            bt.logging.info("Adding timestamp column to database_version table")
                            cursor.execute("ALTER TABLE database_version ADD COLUMN timestamp TEXT")
                            cursor.execute("UPDATE database_version SET timestamp = datetime('now')")
                            conn.commit()
                        
                        bt.logging.info("Fetching current version")
                        cursor.execute("SELECT version FROM database_version ORDER BY timestamp DESC, version DESC LIMIT 1")
                        result = cursor.fetchone()
                        current_version = result[0] if result else '0.0.0'
                        bt.logging.info(f"Current database version: {current_version}")
                    
                    bt.logging.info(f"Comparing current version {current_version} with target version {self.target_version}")
                    if version.parse(current_version) < version.parse(self.target_version):
                        bt.logging.info(f"Database needs migration from {current_version} to {self.target_version}")
                        if not migrate_database(conn, self.db_path, self.target_version):
                            raise Exception("Database migration failed")
                        
                        bt.logging.info("Updating database_version table after migration")
                        cursor.execute("INSERT INTO database_version (version, timestamp) VALUES (?, datetime('now'))", (self.target_version,))
                        conn.commit()
                        bt.logging.info(f"Database version updated to {self.target_version}")
                    else:
                        bt.logging.info(f"Database is already at version {current_version}, no migration needed")
                    
                    bt.logging.info("Creating tables if they don't exist")
                    self._create_tables(cursor)
                    conn.commit()
                bt.logging.info("Database initialization completed successfully")
                
                # Double-check the version after initialization
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT version FROM database_version ORDER BY timestamp DESC, version DESC LIMIT 1")
                    result = cursor.fetchone()
                    bt.logging.info(f"Final database version check: {result[0] if result else 'No version found'}")
                
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    bt.logging.warning(f"Database is locked. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    bt.logging.error(f"Failed to initialize database: {e}")
                    raise
            except Exception as e:
                bt.logging.error(f"Failed to initialize database: {e}")
                raise

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
                miner_uid TEXT,
                miner_cash REAL,
                miner_lifetime_earnings REAL,
                miner_lifetime_wager REAL,
                miner_current_incentive REAL,
                miner_rank INTEGER,
                miner_lifetime_predictions INTEGER,
                miner_lifetime_wins INTEGER,
                miner_lifetime_losses INTEGER,
                miner_win_loss_ratio REAL,
                miner_last_prediction_date TEXT,
                last_daily_reset TEXT
            )
        """)

    @classmethod
    def get_instance(cls, db_path, max_connections=10):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path, max_connections)
        return cls._instance

    @contextmanager
    def get_cursor(self):
        connection = self.connection_pool.get()
        try:
            cursor = connection.cursor()
            yield cursor
            connection.commit()
        except Exception as e:
            bt.logging.error(f"Database error: {e}")
            connection.rollback()
            raise
        finally:
            self.connection_pool.put(connection)

    def close_all(self):
        while not self.connection_pool.empty():
            conn = self.connection_pool.get()
            conn.close()

def get_db_manager(max_connections=10, state_manager=None, miner_uid=None):
    config = bt.config()
    db_path = getattr(config, 'db_path', None)
    
    if db_path is None:
        bt.logging.warning("db_path not found in config. Using default path.")
        # Use the same relative path as in the DatabaseManager __init__ method
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'miner.db')
    
    bt.logging.info(f"Using database path: {db_path}")
    
    
    
    return DatabaseManager.get_instance(db_path, max_connections)