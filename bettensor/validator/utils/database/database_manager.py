import sqlite3
import threading
import bittensor as bt
from queue import Queue, Empty
from bettensor.validator.utils.database.database_init import initialize_database
import logging
import time

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

        bt.logging.info("Starting DatabaseManager initialization...")
        
        # Initialize thread lock
        bt.logging.info("Initializing thread lock...")
        self.lock = threading.Lock()
        
        # Initialize other attributes
        bt.logging.info("Initializing attributes...")
        self.queue = Queue()
        self.db_path = db_path
        self.transaction_active = False
        self.logger = None

        # Setup logging
        bt.logging.info("Setting up logging...")
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        # Initialize thread local storage
        self._thread_local = threading.local()

        # Mark as initialized
        bt.logging.info("DatabaseManager initialization complete.")
        self._initialized = True

        self.worker_thread = None
        self.running = True
        self._start_worker()

    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._thread_local, 'conn') or self._thread_local.conn is None:
            self._thread_local.conn = sqlite3.connect(self.db_path)
            self._thread_local.cursor = self._thread_local.conn.cursor()
            bt.logging.debug(f"Created new database connection for thread {threading.get_ident()}")
        return self._thread_local.conn, self._thread_local.cursor

    def execute_query(self, query, params=None, max_retries=3, initial_delay=0.1):
        attempt = 0
        while attempt < max_retries:
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        conn.commit()
                        return cur.rowcount
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    attempt += 1
                    if attempt == max_retries:
                        raise
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    bt.logging.warning(f"Database locked, retrying in {delay:.2f}s (attempt {attempt}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise

    def begin_transaction(self):
        """Begin a database transaction."""
        with self.lock:
            try:
                conn, _ = self._get_connection()
                if not self.transaction_active:
                    conn.execute("BEGIN")
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
        """Commit the current database transaction."""
        with self.lock:
            try:
                conn, _ = self._get_connection()
                if self.transaction_active:
                    conn.commit()
                    self.transaction_active = False
                    self.logger.debug("Transaction committed.")
                else:
                    self.logger.warning("No active transaction to commit.")
            except sqlite3.Error as e:
                self.logger.error(f"Error committing transaction: {e}")
                raise

    def rollback_transaction(self):
        """Rollback the current database transaction."""
        with self.lock:
            try:
                conn, _ = self._get_connection()
                if self.transaction_active:
                    conn.rollback()
                    self.transaction_active = False
                    self.logger.debug("Transaction rolled back.")
                else:
                    self.logger.warning("No active transaction to rollback.")
            except sqlite3.Error as e:
                self.logger.error(f"Error rolling back transaction: {e}")
                raise

    def _start_worker(self):
        """Start the background worker thread for executing queued database operations."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

    def _worker(self):
        """Background worker thread that processes database operations from the queue."""
        while self.running:
            try:
                # Shorter timeout to allow for graceful shutdown
                func, args, kwargs, result_queue = self.queue.get(timeout=0.1)
                
                with self.lock:
                    try:
                        # Create a new connection for each operation
                        conn = sqlite3.connect(self.db_path, timeout=30)
                        cursor = conn.cursor()
                        
                        try:
                            if isinstance(args[0], str):  # If it's a SQL query
                                cursor.execute(*args)
                                result = cursor.fetchall()
                                conn.commit()
                            else:
                                result = func(*args, **kwargs)
                            
                            if result_queue is not None:
                                result_queue.put(result)
                                
                        except Exception as e:
                            bt.logging.error(f"Database operation error: {str(e)}")
                            if result_queue is not None:
                                result_queue.put(e)
                            conn.rollback()
                            
                        finally:
                            cursor.close()
                            conn.close()
                            
                    except sqlite3.Error as e:
                        bt.logging.error(f"SQLite error: {str(e)}")
                        if result_queue is not None:
                            result_queue.put(e)
                            
            except Empty:
                continue
            except Exception as e:
                bt.logging.error(f"Worker thread error: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on persistent errors

    def execute(self, query, params=None):
        """Execute a query asynchronously through the queue with improved timeout handling."""
        if not self.worker_thread.is_alive():
            bt.logging.warning("Database worker thread died, restarting...")
            self._start_worker()
            
        result_queue = Queue()
        self.queue.put((self.execute_query, (query, params), {}, result_queue))
        
        try:
            # Increased timeout and added retry logic
            for attempt in range(3):  # Try up to 3 times
                try:
                    result = result_queue.get(timeout=10)  # 10 second timeout per attempt
                    if isinstance(result, Exception):
                        raise result
                    return result
                except Empty:
                    if attempt < 2:  # Don't sleep on last attempt
                        time.sleep(1)  # Wait before retry
                    continue
                    
            raise TimeoutError("Database operation timed out after 3 attempts")
            
        except Exception as e:
            bt.logging.error(f"Database execution error: {str(e)}")
            raise

    def execute_query(self, query, params=None):
        """Execute a single query with parameters."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            result = cursor.fetchall()
            conn.commit()
            return result
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
            
        finally:
            if conn:
                conn.close()

    def shutdown(self):
        """Gracefully shutdown the database manager."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def __del__(self):
        """Cleanup thread-local connections on deletion."""
        if hasattr(self, '_thread_local'):
            if hasattr(self._thread_local, 'conn') and self._thread_local.conn:
                self._thread_local.conn.close()

    def is_transaction_active(self):
        """
        Check if a transaction is currently active.

        Returns:
            bool: True if a transaction is active, False otherwise.
        """
        return self.transaction_active

    def executemany(self, query, params):
        """
        Execute a query with multiple parameter sets.
        
        Args:
            query (str): The SQL query to execute
            params (list): List of parameter tuples
            
        Returns:
            int: Number of rows affected
        """
        with self.lock:
            try:
                conn, cursor = self._get_connection()
                cursor.executemany(query, params)
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                self.logger.error(f"Database error in executemany: {e}")
                conn.rollback()
                raise

    def fetch_all(self, query, params=None):
        """
        Fetch all records from the database.
        
        Args:
            query (str): The SQL query to execute
            params (tuple, optional): Parameters for the SQL query
            
        Returns:
            list: List of dictionaries containing the query results
        """
        with self.lock:
            try:
                conn, cursor = self._get_connection()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                results = cursor.fetchall()
                
                # If no results, return empty list
                if not results:
                    return []
                    
                # Convert results to list of dictionaries
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in results]
                
            except sqlite3.Error as e:
                self.logger.error(f"Database error in fetch_all: {e}")
                raise

    def fetch_one(self, query, params=None):
        """
        Fetch a single record from the database.
        
        Args:
            query (str): The SQL query to execute
            params (tuple, optional): Parameters for the SQL query
            
        Returns:
            dict or int: Dictionary containing the query results, or integer for COUNT queries
        """
        with self.lock:
            try:
                conn, cursor = self._get_connection()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                result = cursor.fetchone()
                
                # If result is None, return None
                if result is None:
                    return None
                    
                # If the query is a COUNT query, return just the count value
                if query.strip().lower().startswith('select count'):
                    return result[0]
                    
                # Convert result to dictionary
                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, result))
                
            except sqlite3.Error as e:
                self.logger.error(f"Database error in fetch_one: {e}")
                raise

    def transaction(self):
        return DatabaseTransaction(self)
    
    def get_connection(self):
        return self._get_connection()

class DatabaseTransaction:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.conn = None
        
    def __enter__(self):
        self.conn = self.db_manager.get_connection()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()