import sqlite3
import threading
import bittensor as bt
from queue import Queue, Empty
from bettensor.validator.utils.database.database_init import initialize_database
import logging


class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path):
        if not hasattr(self, "initialized"):
            self.db_path = db_path
            self.connection = None
            self.cursor = None
            self.connect()
            self.check_and_migrate_schema()
            self.transaction_active = False
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)
            self.queue = Queue()
            self.lock = threading.Lock()
            self._initialize_database()
            self._start_worker()
            self.initialized = True

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Debug: Successfully connected to database at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def begin_transaction(self):
        with self.lock:
            if not self.transaction_active:
                self.conn.execute("BEGIN")
                self.transaction_active = True
                self.logger.debug("Transaction started.")
            else:
                self.logger.warning("Transaction already active.")

    def commit_transaction(self):
        with self.lock:
            if self.transaction_active:
                self.conn.commit()
                self.transaction_active = False
                self.logger.debug("Transaction committed.")
            else:
                self.logger.warning("No active transaction to commit.")

    def rollback_transaction(self):
        with self.lock:
            if self.transaction_active:
                self.conn.rollback()
                self.transaction_active = False
                self.logger.debug("Transaction rolled back.")
            else:
                self.logger.warning("No active transaction to rollback.")

    def _start_worker(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            try:
                func, args, kwargs, result_queue = self.queue.get(timeout=1)
                with self.lock:
                    try:
                        if callable(func):
                            result = func(*args, **kwargs)
                        else:
                            raise TypeError("The 'func' argument must be callable")
                        if result_queue is not None:
                            result_queue.put(result)
                    except Exception as e:
                        if result_queue is not None:
                            result_queue.put(e)
                        bt.logging.error(f"Error in database operation: {str(e)}")
            except Empty:
                continue

    def execute(self, query, params=None, func=None):
        """
        Execute a database operation asynchronously.

        Args:
            query (str): The SQL query to execute.
            params (tuple, list, or list of tuples): Parameters for the SQL query.
            func (callable): The function to execute with the cursor.
        """
        if func is not None and not callable(func):
            raise TypeError("The 'func' argument must be callable")

        result_queue = Queue()

        def _execute():
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

        self.queue.put((_execute, (), {}, result_queue))
        return result_queue

    def execute_query(self, query, params=None):
        with self.lock:
            try:
                if params:
                    self.cursor.execute(query, params)
                else:
                    self.cursor.execute(query)
                self.conn.commit()
                # self.logger.debug(f"Query executed successfully: {query}")
                return self.cursor.rowcount
            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                self.conn.rollback()
                raise

    def executemany(self, query, params):
        with self.lock:
            try:
                self.cursor.executemany(query, params)
                self.conn.commit()
                # self.logger.debug(f"Executemany query executed successfully. Rows affected: {self.cursor.rowcount}")
                return self.cursor.rowcount
            except sqlite3.Error as e:
                self.logger.error(f"Database error in executemany: {e}")
                self.conn.rollback()
                raise

    def fetch_one(self, query, params=None):
        with self.lock:
            try:
                if params:
                    self.cursor.execute(query, params)
                else:
                    self.cursor.execute(query)
                result = self.cursor.fetchone()
                return dict(result) if result else None
            except sqlite3.Error as e:
                self.logger.error(f"Database error in fetch_one: {e}")
                raise

    def fetch_all(self, query, params=None):
        with self.lock:
            try:
                if params:
                    self.cursor.execute(query, params)
                else:
                    self.cursor.execute(query)
                results = self.cursor.fetchall()
                return [dict(row) for row in results]
            except sqlite3.Error as e:
                self.logger.error(f"Database error in fetch_all: {e}")
                raise

    def _initialize_database(self):
        queries = initialize_database()
        with self.lock:
            for query in queries:
                self.cursor.execute(query)
            self.conn.commit()
        self.logger.debug("Database tables created successfully")

    def __del__(self):
        if self.conn:
            self.conn.close()

    def is_transaction_active(self):
        return self.transaction_active

    def check_and_migrate_schema(self):
        """Check database version and perform migrations if needed"""
        try:
            # Check if version table exists
            version_exists = self.fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='db_version'"
            )
            
            if not version_exists:
                # Create version table
                self.execute_query(
                    "CREATE TABLE db_version (version INTEGER PRIMARY KEY)"
                )
                self.execute_query("INSERT INTO db_version (version) VALUES (0)")
                current_version = 0
            else:
                current_version = self.fetch_one(
                    "SELECT version FROM db_version"
                )['version']

            # If we're on version 0, migrate to version 1
            if current_version < 1:
                bt.logging.info("Migrating database schema to version 1...")
                
                # Begin transaction
                self.begin_transaction()
                
                try:
                    # Run migration queries
                    migration_queries = initialize_database()
                    for query in migration_queries:
                        self.execute_query(query)
                    
                    # Update version
                    self.execute_query(
                        "UPDATE db_version SET version = 1"
                    )
                    
                    # Commit transaction
                    self.commit_transaction()
                    bt.logging.info("Database migration completed successfully")
                    
                except Exception as e:
                    self.rollback_transaction()
                    bt.logging.error(f"Database migration failed: {e}")
                    raise
                    
        except Exception as e:
            bt.logging.error(f"Error checking/migrating database schema: {e}")
            raise