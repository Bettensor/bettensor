import sqlite3
from contextlib import contextmanager
from threading import Lock
import os
import bittensor as bt
from sqlite3 import OperationalError
import time

class DatabaseManager:
    _instances = {}

    @classmethod
    def get_instance(cls, db_path):
        bt.logging.trace(f"get_instance() | Getting instance for {db_path}")
        if db_path not in cls._instances:
            cls._instances[db_path] = cls(db_path)
        return cls._instances[db_path]

    def __init__(self, db_path):
        bt.logging.trace(f"__init__() | Initializing database manager for {db_path}")
        self.db_path = db_path
        self.lock = Lock()
        self.connection_pool = []

    def get_connection(self):
        bt.logging.trace(f"get_connection() | Getting connection for {self.db_path}")
        with self.lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            else:
                return sqlite3.connect(self.db_path, check_same_thread=False)

    def release_connection(self, conn):
        with self.lock:
            self.connection_pool.append(conn)

    @contextmanager
    def get_cursor(self):
        bt.logging.trace(f"get_cursor() | Getting cursor for {self.db_path}")
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except OperationalError as e:
            bt.logging.error(f"Database error: {e}. Retrying...")
            time.sleep(1)  # Wait for a second before retrying
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        finally:
            cursor.close()
            self.release_connection(conn)

def get_db_manager(miner_uid=None):
    '''
    This function returns a database manager instance.
    If USE_SINGLE_DB is set to true, it points all miners to the same database.
    If USE_SINGLE_DB is set to false, it creates a new database for each miner.

    This can help with concurrency issues, if a user is running many miners at once. If they are just running 1-3 miners,
    it is recommended to set USE_SINGLE_DB to true. If they are running many miners, it is recommended to set USE_SINGLE_DB to false.
    '''
    bt.logging.trace(f"get_db_manager() | Getting database manager for {miner_uid}")

    use_single_db = os.environ.get('USE_SINGLE_DB', 'True').lower() == 'true'
    bt.logging.trace(f"get_db_manager() | Using single database: {use_single_db}")
    
    if use_single_db:
        db_path = os.environ.get('SINGLE_DB_PATH', './data/miner.db')
    else:
        if miner_uid is None:
            db_path = os.environ.get('DEFAULT_DB_PATH', './data/miner.db')
        else:
            db_path = os.environ.get(f'MINER_{miner_uid}_DB_PATH', f'./data/miner_{miner_uid}.db')
    
    return DatabaseManager.get_instance(db_path)