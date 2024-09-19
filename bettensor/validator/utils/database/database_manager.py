import sqlite3
import threading
from queue import Queue, Empty
from bettensor.validator.utils.database.database_init import initialize_database


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
            self.queue = Queue()
            self.lock = threading.Lock()
            self._start_worker()
            self._initialize_database()
            self.initialized = True

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
                        result = func(cursor, *args, **kwargs)
                        if func.__name__ not in [
                            "begin_transaction",
                            "rollback_transaction",
                        ]:
                            conn.commit()
                        result_queue.put(result)
                    except Exception as e:
                        conn.rollback()
                        result_queue.put(e)
                    finally:
                        conn.close()
            except Empty:
                continue

    def execute(self, func, *args, **kwargs):
        result_queue = Queue()
        self.queue.put((func, args, kwargs, result_queue))
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result

    def fetchall(self, query, params=()):
        return self.execute_query(query, params, fetch=True)

    def fetchone(self, query, params=()):
        result = self.execute_query(query, params, fetch=True)
        return result[0] if result else None

    def execute_query(self, query, params=(), fetch=False):
        def _execute_query(cursor):
            cursor.execute(query, params or ())
            if fetch:
                return cursor.fetchall()
            else:
                return cursor.rowcount  # Return the number of affected rows

        return self.execute(_execute_query)

    def executemany(self, query, params_list):
        def _executemany(cursor):
            cursor.executemany(query, params_list)

        self.execute(_executemany)

    def begin_transaction(self):
        def _begin_transaction(cursor):
            cursor.execute("BEGIN TRANSACTION")

        self.execute(_begin_transaction)

    def commit_transaction(self):
        def _commit_transaction(cursor):
            cursor.execute("COMMIT")

        self.execute(_commit_transaction)

    def rollback_transaction(self):
        def _rollback_transaction(cursor):
            cursor.execute("ROLLBACK")

        self.execute(_rollback_transaction)

    def _initialize_database(self):
        queries = initialize_database()
        for query in queries:
            self.execute_query(query)
