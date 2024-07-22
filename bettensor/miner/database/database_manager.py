import sqlite3
from contextlib import contextmanager
from threading import Lock
import os
import bittensor as bt
from sqlite3 import OperationalError
import time
import datetime
from typing import Dict, Tuple
from bettensor.protocol import TeamGame

# Import operation classes 
from .games import Games
from .predictions import Predictions
from bettensor.miner.stats import MinerStatsHandler

class DatabaseManager:
    _instance = None
    _lock = Lock()
    _connection_pool = []
    _max_connections = 10  # Increased default

    def __init__(self, db_path, max_connections=None):
        bt.logging.trace(f"Initializing DatabaseManager with db_path: {db_path}, max_connections: {max_connections}")
        if not DatabaseManager._instance:
            bt.logging.trace(f"__init__() | Initializing database manager for {db_path}")
            self.db_path = db_path
            
            if max_connections is not None:
                DatabaseManager._max_connections = max_connections
            
            # Ensure the database directory exists and initialize the database
            self.ensure_db_directory_exists()
            self.initialize_database()
            
            # Initialize operation classes
            self.games = Games(self)
            self.predictions = Predictions(self)
            self.miner_stats = MinerStatsHandler(self)

            DatabaseManager._instance = self
        bt.logging.trace("DatabaseManager initialization complete")

    @classmethod
    def get_instance(cls, max_connections=None):
        bt.logging.trace(f"Getting DatabaseManager instance with max_connections: {max_connections}")
        if cls._instance is None:
            db_path = os.environ.get('DB_PATH', './data/miner.db')
            with cls._lock:
                if cls._instance is None:  # Double-check locking
                    cls._instance = cls(db_path, max_connections)
        bt.logging.trace("DatabaseManager instance retrieved")
        return cls._instance

    @classmethod
    def get_connection(cls):
        bt.logging.trace("Getting database connection")
        with cls._lock:
            if cls._connection_pool:
                return cls._connection_pool.pop()
            else:
                conn = sqlite3.connect(cls._instance.db_path, check_same_thread=False)
                conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL mode
                bt.logging.trace("Database connection obtained")
                return conn

    @classmethod
    def release_connection(cls, conn):
        bt.logging.trace("Releasing database connection")
        with cls._lock:
            if len(cls._connection_pool) < cls._max_connections:
                cls._connection_pool.append(conn)
            else:
                conn.close()
        bt.logging.trace("Database connection released")

    @classmethod
    @contextmanager
    def get_cursor(cls):
        bt.logging.trace("Getting database cursor")
        conn = cls.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
            bt.logging.trace("Database operation completed successfully")
        except OperationalError as e:
            bt.logging.error(f"Database error: {e}. Retrying...")
            time.sleep(1)  # Wait for a second before retrying
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        finally:
            cursor.close()
            cls.release_connection(conn)

    def ensure_db_directory_exists(self) -> bool:
        bt.logging.trace(f"Ensuring database directory exists for {self.db_path}")
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        bt.logging.trace("Database directory check complete")

    def initialize_database(self):
        bt.logging.trace(f"Initializing database at {self.db_path}")
        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS predictions (
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
                                   canOverwrite BOOLEAN,
                                   outcome TEXT
                                   )"""
                )
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS games (
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
                                   )"""
                )
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS miner_stats (
                                   minerID TEXT PRIMARY KEY, 
                                   balance REAL,
                                   last_update_time TEXT
                                   )"""
                )
        except sqlite3.Error as e:
            bt.logging.error(f"Failed to initialize local database: {e}")
            raise Exception("Failed to initialize local database")
        bt.logging.trace("Database initialization complete")

    # Game operations
    def add_game_data(self, game_data):
        return self.game_ops.add_game_data(game_data)

    def update_game_data(self, game_data):
        return self.game_ops.update_game_data(game_data)

    def get_games(self, filters=None):
        return self.game_ops.get_games(filters)

    # Prediction operations
    def add_prediction(self, prediction_data):
        return self.prediction_ops.add_prediction(prediction_data)

    def get_predictions(self, filters=None):
        return self.prediction_ops.get_predictions(filters)

def get_db_manager(max_connections=None):
    bt.logging.trace(f"Getting DatabaseManager with max_connections: {max_connections}")
    return DatabaseManager.get_instance(max_connections)