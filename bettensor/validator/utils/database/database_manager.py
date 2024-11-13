import asyncio
from pathlib import Path
from sqlite3 import OperationalError
import sqlite3
import time
import aiosqlite
import bittensor as bt
import os
import traceback
from bettensor.validator.utils.database.database_init import initialize_database
import async_timeout

class DatabaseManager:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, db_path):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._instance_lock = asyncio.Lock()
        return cls._instance

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self._transaction_in_progress = False
        self._lock = asyncio.Lock()
        self._connection_lock = asyncio.Lock()
        self._pause_lock = asyncio.Lock()
        self._operations_paused = False
        self._transaction_timeout = 120  # 30 second timeout

    async def ensure_connection(self):
        """Ensures database connection exists and is valid"""
        if self._operations_paused:
            return False
            
        async with self._connection_lock:
            try:
                if self.conn is None:
                    self.conn = await aiosqlite.connect(self.db_path)
                    await self.conn.execute("PRAGMA journal_mode=WAL")
                    await self.conn.execute("PRAGMA busy_timeout=5000")
                    bt.logging.debug("Database connection established")
                
                # Test the connection
                async with self.conn.execute("SELECT 1") as cursor:
                    await cursor.fetchone()
                return True
                
            except Exception as e:
                bt.logging.error(f"Database connection error: {str(e)}")
                if self.conn:
                    await self.conn.close()
                self.conn = None
                return False

    async def get_connection(self):
        """Get database connection, creating it if needed."""
        if self.conn is None:
            self.conn = await aiosqlite.connect(self.db_path)
        return self.conn

    async def initialize(self):
        async with self._instance_lock:
            if self._initialized:
                return
            bt.logging.info("Initializing DatabaseManager...")
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = await aiosqlite.connect(self.db_path)
            await self.reset_database_pragmas()
            
            # Initialize the database schema
            statements = initialize_database()
            for statement in statements:
                try:
                    await self.conn.execute(statement.strip())
                    await self.conn.commit()  # Commit after each statement
                except sqlite3.OperationalError as e:
                    # Ignore specific errors for ALTER TABLE statements
                    if "duplicate column" in str(e):
                        bt.logging.debug(f"Column already exists, skipping: {str(e)}")
                        continue
                    else:
                        bt.logging.error(f"Error executing statement: {e}")
                        bt.logging.error(f"Failed statement: {statement}")
                        raise
                except Exception as e:
                    bt.logging.error(f"Error executing statement: {e}")
                    bt.logging.error(f"Failed statement: {statement}")
                    raise
                    
            self._initialized = True
            bt.logging.info("DatabaseManager initialization complete.")

    async def reset_database_pragmas(self):
        """Reset database PRAGMAs to default values asynchronously."""
        try:
            await self.conn.execute("PRAGMA journal_mode = WAL")
            await self.conn.execute("PRAGMA synchronous = NORMAL")
            await self.conn.execute("PRAGMA foreign_keys = ON")
            await self.conn.execute("PRAGMA cache_size = -2000")
            await self.conn.commit()
        except Exception as e:
            bt.logging.error(f"Error resetting database PRAGMAs: {e}")
            raise

    async def execute_query(self, query, params=None):
        """Execute query with pause check and timeout"""
        if self._operations_paused:
            bt.logging.warning("Database operations are paused - query rejected")
            raise OperationalError("Database operations are paused")
            
        if not await self.ensure_connection():
            raise ConnectionError("No active database connection")
            
        try:
            async with async_timeout.timeout(self._transaction_timeout):
                async with self._lock:
                    return await self.conn.execute(query, params if params else ())
        except asyncio.TimeoutError:
            bt.logging.error(f"Query execution timed out: {query[:100]}...")
            if self._lock.locked():
                self._lock.release()
            raise
        except Exception as e:
            bt.logging.error(f"Query execution error: {str(e)}")
            if self._lock.locked():
                self._lock.release()
            raise

    async def execute_state_sync_query(self, query, params=None, max_retries=3, retry_delay=5):
        """Special query execution for state sync operations with retry logic"""
        if not self.conn:
            if not await self.ensure_connection():
                raise ConnectionError("No active database connection")
                
        for attempt in range(max_retries):
            try:
                async with self._lock:
                    cursor = await self.conn.execute(query, params if params else ())
                    await self.conn.commit()
                    return cursor
            except sqlite3.OperationalError as e:
                if "database table is locked" in str(e):
                    if attempt < max_retries - 1:
                        bt.logging.warning(f"Database locked, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        continue
                raise
            except Exception as e:
                bt.logging.error(f"State sync query execution error: {str(e)}")
                raise

    async def fetch_all(self, query, params=None):
        """Execute a SELECT query and return all results as dictionaries"""
        if not await self.ensure_connection():
            return None
            
        try:
            async with self._lock:
                async with self.conn.execute(query, params if params else ()) as cursor:
                    # Get column names
                    columns = [description[0] for description in cursor.description]
                    # Fetch all rows and convert to dictionaries
                    rows = await cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            bt.logging.error(f"Error executing query: {str(e)}")
            return None

    async def fetch_one(self, query, params=None):
        """Fetch a single record from the database asynchronously."""
        async with self._lock:
            try:
                # Ensure connection exists
                if self.conn is None:
                    self.conn = await self.get_connection()
                    if self.conn is None:
                        raise RuntimeError("Failed to initialize database connection")
                
                async with self.conn.execute(query, params or ()) as cursor:
                    row = await cursor.fetchone()
                    if row is None:
                        return None
                    columns = [column[0] for column in cursor.description]
                    return dict(zip(columns, row))
            except Exception as e:
                bt.logging.error(f"Database error in fetch_one: {e}")
                raise

    async def executemany(self, query, params_list):
        """Execute many queries asynchronously."""
        async with self._lock:
            try:
                await self.conn.executemany(query, params_list)
                await self.conn.commit()
            except Exception as e:
                bt.logging.error(f"Error in executemany: {e}")
                raise

    async def begin_transaction(self):
        """Begin a new transaction with connection check and timeout"""
        if not await self.ensure_connection():
            raise ConnectionError("Could not establish database connection")
            
        try:
            # Try to acquire lock with timeout
            async with async_timeout.timeout(self._transaction_timeout):
                await self._lock.acquire()
                
                if self._transaction_in_progress:
                    self._lock.release()
                    raise RuntimeError("Transaction already in progress")
                    
                await self.conn.execute("BEGIN")
                self._transaction_in_progress = True
                
        except asyncio.TimeoutError:
            if self._lock.locked():
                self._lock.release()
            raise TimeoutError("Transaction lock acquisition timed out")
        except Exception as e:
            if self._lock.locked():
                self._lock.release()
            raise

    async def commit_transaction(self):
        """Commit transaction and release lock"""
        if not self.conn:
            raise ConnectionError("No active database connection")
            
        if not self._transaction_in_progress:
            return
            
        try:
            await self.conn.commit()
        finally:
            self._transaction_in_progress = False
            if self._lock.locked():
                self._lock.release()
            bt.logging.debug("Transaction committed and lock released")

    async def rollback_transaction(self):
        """Rollback transaction and release lock"""
        if self._operations_paused:
            bt.logging.warning("Cannot rollback - database operations are paused")
            return
            
        if not await self.ensure_connection():
            return
            
        if not self._transaction_in_progress:
            return
            
        try:
            await self.conn.rollback()
        finally:
            self._transaction_in_progress = False
            if self._lock.locked():
                self._lock.release()

    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def cleanup(self):
        """Cleanup database connections with retry logic"""
        bt.logging.info("Cleaning up database connections...")
        try:
            await self.pause_operations()
            await asyncio.sleep(2)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with self._lock:
                        await self.execute_state_sync_query("PRAGMA wal_checkpoint(TRUNCATE);")
                        await asyncio.sleep(1)
                        
                        if self.conn:
                            await self.conn.close()
                            self.conn = None
                            
                        bt.logging.info("Database cleanup completed successfully")
                        break
                        
                except sqlite3.OperationalError as e:
                    if "database table is locked" in str(e) and attempt < max_retries - 1:
                        bt.logging.warning(f"Cleanup retry {attempt + 1}/{max_retries} - waiting for locks to clear")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise
                    
        except Exception as e:
            bt.logging.error(f"Error during database cleanup: {str(e)}")
            if self.conn:
                try:
                    await self.conn.close()
                except:
                    pass
            self.conn = None
        finally:
            self._operations_paused = False

    async def pause_operations(self):
        """Pause database operations after ensuring all pending operations are complete."""
        max_wait = 120  # Maximum seconds to wait for operations to complete
        start_time = time.time()
        
        while await self.has_pending_operations():
            if time.time() - start_time > max_wait:
                bt.logging.warning("Timeout waiting for database operations to complete")
                break
            await asyncio.sleep(1)
        
        self._operations_paused = True
        bt.logging.info("Database operations paused")
    
    async def resume_operations(self):
        """Resume database operations"""
        async with self._lock:
            self._operations_paused = False
            bt.logging.info("Database operations resumed")
    
    async def reconnect(self):
        """Force a reconnection to the database"""
        async with self._connection_lock:
            if self.conn:
                try:
                    await self.conn.close()
                except Exception:
                    pass
            self.conn = None
            self._transaction_in_progress = False
            return await self.ensure_connection()

    async def has_pending_operations(self) -> bool:
        """Check if there are any pending database operations."""
        try:
            # Check WAL file size
            wal_path = Path(self.db_path).with_suffix('.db-wal')
            if wal_path.exists() and wal_path.stat().st_size > 0:
                return True
            
            # Check for active transactions
            result = await self.fetch_all("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' 
                AND name='sqlite_stat1'
                AND EXISTS (
                    SELECT 1 FROM sqlite_master WHERE type='temp_table'
                )
            """)
            return result[0][0] > 0
        except Exception as e:
            bt.logging.error(f"Error checking pending operations: {e}")
            return True  # Assume there are pending operations if we can't check

# Usage example
async def main():
    db_path = "path/to/your/database.db"
    db_manager = DatabaseManager(db_path)
    await db_manager.initialize()

    # Example query
    result = await db_manager.execute_query("SELECT * FROM your_table")
    print(result)

# Run the example
if __name__ == "__main__":
    asyncio.run(main())

