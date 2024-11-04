import sqlite3
import threading
import bittensor as bt
from queue import Queue, Empty
from bettensor.validator.utils.database.database_init import initialize_database
import logging

class SingletonMeta(type):
    """
    A thread-safe implementation of Singleton using a metaclass.
    """
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking to ensure thread-safe singleton
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class DatabaseManager(metaclass=SingletonMeta):
    """
    Singleton DatabaseManager to handle database connections and migrations.
    """

    def __init__(self, db_path):
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            bt.logging.info("DatabaseManager already initialized, skipping initialization.")
            return

        # Initialize thread lock
        bt.logging.info("Initializing DatabaseManager thread lock.")
        self.lock = threading.Lock()

        # Initialize other attributes
        self.queue = Queue()
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.transaction_active = False
        self.logger = None

        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        # Establish database connection
        self.connect()

        # Initialize database schema
        self._initialize_database()

        # Check and migrate schema if necessary
        self.check_and_migrate_schema()

        # Start background worker thread
        self._start_worker()

        # Mark as initialized
        self._initialized = True

    def connect(self):
        """
        Establish a connection to the SQLite database.
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            bt.logging.info(f"Successfully connected to database at {self.db_path}")
        except sqlite3.Error as e:
            bt.logging.error(f"Error connecting to database: {e}")
            raise

    def _initialize_database(self):
        """
        Initialize the database with required tables.
        """
        queries = initialize_database()
        with self.lock:
            for query in queries:
                self.cursor.execute(query)
            self.conn.commit()
        self.logger.debug("Database tables created successfully")

    def check_and_migrate_schema(self):
        """
        Check the current database schema version and perform migrations if needed.
        """
        try:
            with self.lock:
                # Check if version table exists
                version_exists = self.fetch_one(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='db_version'"
                )

                if not version_exists:
                    self.execute_query("CREATE TABLE db_version (version INTEGER PRIMARY KEY)")
                    self.execute_query("INSERT INTO db_version (version) VALUES (0)")
                    current_version = 0
                else:
                    version = self.fetch_one("SELECT version FROM db_version")
                    current_version = version['version'] if version else 0

            # Example migration from version 0 to 1
            if current_version < 1:
                bt.logging.info("Migrating database schema to version 1...")
                self.begin_transaction()
                try:
                    # Add migration queries here
                    # Example:
                    # self.execute_query("ALTER TABLE some_table ADD COLUMN new_column TEXT")

                    self.execute_query("UPDATE db_version SET version = 1")
                    
                    self.commit_transaction()
                    bt.logging.info("Database migration to version 1 completed successfully.")
                except Exception as e:
                    self.rollback_transaction()
                    bt.logging.error(f"Database migration failed: {e}")
                    raise
        except Exception as e:
            bt.logging.error(f"Error checking/migrating database schema: {e}")
            raise

    def fetch_one(self, query, params=None):
        """
        Fetch a single record from the database.
        """
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

    def execute_query(self, query, params=None):
        """
        Execute a single SQL query.
        """
        with self.lock:
            try:
                if params:
                    self.cursor.execute(query, params)
                else:
                    self.cursor.execute(query)
                self.conn.commit()
            except sqlite3.Error as e:
                self.logger.error(f"Database error in execute_query: {e}")
                self.conn.rollback()
                raise

    def begin_transaction(self):
        """
        Begin a database transaction.
        """
        with self.lock:
            try:
                if not self.transaction_active:
                    self.conn.execute("BEGIN")
                    self.transaction_active = True
                    self.logger.debug("Transaction started.")
                else:
                    self.logger.debug("Transaction already active, continuing with existing transaction.")
            except sqlite3.OperationalError as e:
                if "within a transaction" in str(e):
                    self.logger.debug("Transaction already active, continuing with existing transaction.")
                else:
                    raise

    def commit_transaction(self):
        """
        Commit the current database transaction.
        """
        with self.lock:
            if self.transaction_active:
                self.conn.commit()
                self.transaction_active = False
                self.logger.debug("Transaction committed.")
            else:
                self.logger.warning("No active transaction to commit.")

    def rollback_transaction(self):
        """
        Rollback the current database transaction.
        """
        with self.lock:
            if self.transaction_active:
                self.conn.rollback()
                self.transaction_active = False
                self.logger.debug("Transaction rolled back.")
            else:
                self.logger.warning("No active transaction to rollback.")

    def _start_worker(self):
        """
        Start the background worker thread for executing queued database operations.
        """
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        """
        Background worker thread that processes database operations from the queue.
        """
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
            params (tuple, optional): Parameters for the SQL query.
            func (callable, optional): A function to execute with the query.

        Returns:
            The result of the database operation or None if it timed out.
        """
        result_queue = Queue()
        self.queue.put(func or self.execute_query, args=(query, params), kwargs={}, result_queue=result_queue)

        try:
            result = result_queue.get(timeout=5)
            if isinstance(result, Exception):
                raise result
            return result
        except Empty:
            self.logger.error("Database operation timed out")
            return None

    def __del__(self):
        """
        Destructor to close the database connection upon deletion.
        """
        if self.conn:
            self.conn.close()

    def is_transaction_active(self):
        """
        Check if a transaction is currently active.

        Returns:
            bool: True if a transaction is active, False otherwise.
        """
        return self.transaction_active