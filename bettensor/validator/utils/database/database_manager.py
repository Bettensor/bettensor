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
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.queue = Queue()
        self.lock = threading.Lock()
        self._start_worker()
        self._initialize_database()
        self.initialized = True
        self._transaction_active = False

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Debug: Successfully connected to database at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def begin_transaction(self):
        self.execute(lambda cursor: cursor.execute("BEGIN"))
        self._transaction_active = True

    def commit_transaction(self):
        if self._transaction_active:
            self.execute(lambda cursor: cursor.execute("COMMIT"))
            self._transaction_active = False
        else:
            bt.logging.warning("No active transaction to commit")

    def rollback_transaction(self):
        if self._transaction_active:
            self.execute(lambda cursor: cursor.execute("ROLLBACK"))
            self._transaction_active = False
        else:
            bt.logging.warning("No active transaction to rollback")

    def _start_worker(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            try:
                func, args, kwargs, result_queue = self.queue.get(timeout=1)
                with self.lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    try:
                        if callable(func):
                            result = func(cursor)
                        else:
                            raise TypeError("The 'func' argument must be callable")
                        conn.commit()
                        if result_queue is not None:
                            result_queue.put(result)
                    except Exception as e:
                        if result_queue is not None:
                            result_queue.put(e)
                        bt.logging.error(f"Error in database operation: {str(e)}")
                    finally:
                        conn.close()
            except Empty:
                continue

    def execute(self, func, *args, **kwargs):
        """
        Execute a database operation asynchronously.

        Args:
            func (callable): The function to execute with the cursor.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        """
        if not callable(func):
            raise TypeError("The 'func' argument must be callable")
        
        result_queue = kwargs.pop('result_queue', None)
        self.queue.put((func, args, kwargs, result_queue))

    def execute_query(self, query, params=None, batch=False):
        """
        Execute a SQL query synchronously.

        Args:
            query (str): The SQL query to execute.
            params (tuple, list, or list of tuples): Parameters for the SQL query.
            batch (bool): Whether to execute as a batch operation.
        """
        def _execute(cursor):
            if batch and isinstance(params, list):
                cursor.executemany(query, params)
            elif params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

        self.queue.put((_execute, (), {}, None))

    def fetch_one(self, query, params):
        """
        Fetch a single record from the database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): Parameters for the SQL query.

        Returns:
            The fetched record.
        """
        result_queue = Queue()

        def _execute_fetch(cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

        self.queue.put((_execute_fetch, (), {}, result_queue))
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result

    def fetch_all(self, query, params):
        """
        Fetch all records from the database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): Parameters for the SQL query.

        Returns:
            List of fetched records.
        """
        result_queue = Queue()

        def _execute_fetch(cursor):
            cursor.execute(query, params)
            return cursor.fetchall()

        self.queue.put((_execute_fetch, (), {}, result_queue))
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result


    def _initialize_database(self):
        queries = initialize_database()
        for query in queries:
            self.execute_query(query)

    def __del__(self):
        if self.conn:
            self.conn.close()

    def is_transaction_active(self):
        return self._transaction_active
