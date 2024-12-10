import asyncio
from pathlib import Path
from sqlite3 import OperationalError
import sqlite3
import time
import aiosqlite
import bittensor as bt
import os
import traceback

import sqlalchemy
from sqlalchemy import inspect
from bettensor.validator.utils.database.database_init import initialize_database
import async_timeout
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager
import re
import itertools
import aiofiles
import hashlib
import json
import random
import weakref

class DatabaseManager:
    _instance = None

    def __new__(cls, db_path):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, database_path: str):
        """Initialize the database manager with WAL mode and optimized concurrency settings"""
        self.database_path = database_path
        self._shutting_down = False
        self._active_sessions = set()
        self._cleanup_event = asyncio.Event()
        self._cleanup_task = None
        self._connection_attempts = 0
        self._max_connection_attempts = 50
        self._connection_attempt_reset = time.time()
        self._connection_reset_interval = 60
        self.default_timeout = 30
        
        # Initialize engine with WAL mode and optimized settings
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{database_path}",
            echo=False,
            connect_args={
                "timeout": 30,  # SQLite busy timeout
                "check_same_thread": False,
                "isolation_level": None,  # Let SQLAlchemy handle transactions
            }
        )
        
        self.async_session = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def _initialize_connection(self, connection):
        """Initialize connection with WAL mode and optimized settings"""
        await connection.execute(text("PRAGMA journal_mode = WAL"))
        await connection.execute(text("PRAGMA synchronous = NORMAL"))
        await connection.execute(text("PRAGMA busy_timeout = 30000"))
        await connection.execute(text("PRAGMA temp_store = MEMORY"))
        await connection.execute(text("PRAGMA cache_size = -2000"))
        await connection.execute(text("PRAGMA mmap_size = 268435456"))
        await connection.execute(text("PRAGMA wal_autocheckpoint = 1000"))
        await connection.execute(text("PRAGMA read_uncommitted = 1"))  # Allow reading uncommitted changes

    @asynccontextmanager
    async def get_session(self):
        """Session context manager with improved error handling and connection management"""
        if self._shutting_down:
            raise RuntimeError("Database manager is shutting down")
            
        session = None
        connection = None
        retries = 3
        last_error = None
        
        for attempt in range(retries):
            try:
                # Use a very short timeout for initial connection attempt
                async with async_timeout.timeout(1):
                    connection = await self._acquire_connection()
                    
                    # Create session with minimal initialization
                    session = self.async_session(bind=connection)
                    session.created_at = time.time()
                    self._active_sessions.add(session)
                    
                    # Only set essential pragmas
                    await session.execute(text("PRAGMA busy_timeout = 5000"))
                    
                    yield session
                    break
                    
            except asyncio.TimeoutError:
                last_error = f"Timeout on attempt {attempt + 1}/{retries}"
                if attempt < retries - 1:
                    # Add exponential backoff with jitter
                    delay = min(0.1 * (2 ** attempt) + random.uniform(0, 0.1), 1.0)
                    bt.logging.debug(f"Connection timeout, retrying in {delay:.2f}s ({attempt + 1}/{retries})")
                    await asyncio.sleep(delay)
                    continue
                bt.logging.error(f"Database timeout after {retries} attempts")
                raise
                
            except Exception as e:
                last_error = str(e)
                if attempt < retries - 1:
                    delay = min(0.1 * (2 ** attempt) + random.uniform(0, 0.1), 1.0)
                    bt.logging.debug(f"Connection error, retrying in {delay:.2f}s ({attempt + 1}/{retries}): {e}")
                    await asyncio.sleep(delay)
                    continue
                bt.logging.error(f"Database error after {retries} attempts: {e}")
                raise
                
            finally:
                if session and session in self._active_sessions:
                    self._active_sessions.remove(session)
                    await self._safe_close_session(session)
                if connection:
                    await self._safe_close_connection(connection)

    async def _acquire_connection(self):
        """Acquire a database connection."""
        try:
            if not self.engine:
                self.engine = create_async_engine(
                    f"sqlite+aiosqlite:///{self.db_path}",
                    echo=False,
                    pool_pre_ping=True,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=3600
                )

            connection = await self.engine.connect()
            await connection.execute(text("PRAGMA busy_timeout = 30000"))  # 30 second timeout
            await connection.execute(text("PRAGMA journal_mode = WAL"))
            await connection.execute(text("PRAGMA synchronous = NORMAL"))
            return connection

        except Exception as e:
            bt.logging.error(f"Error acquiring connection: {str(e)}")
            raise

    async def _cleanup_stale_connections(self):
        """Cleanup stale connections and sessions"""
        try:
            current_time = time.time()
            stale_timeout = 60  # Reduced from 300
            
            # Cleanup stale sessions
            stale_sessions = [
                session for session in self._active_sessions
                if hasattr(session, 'created_at') and 
                current_time - session.created_at > stale_timeout
            ]
            
            for session in stale_sessions:
                await self._safe_close_session(session)
                
            # Periodically dispose engine connections
            if (current_time - getattr(self, '_last_engine_cleanup', 0)) > 300:  # Every 5 minutes
                await self.engine.dispose()
                self._last_engine_cleanup = current_time
                
        except Exception as e:
            bt.logging.error(f"Error in connection cleanup: {e}")

    async def _safe_close_connection(self, connection):
        """Safely close a connection with timeout"""
        if connection:
            try:
                async with async_timeout.timeout(1):
                    await connection.close()
            except Exception as e:
                bt.logging.debug(f"Error closing connection: {e}")

    async def cleanup(self):
        """Clean shutdown of database connections"""
        try:
            self._shutting_down = True
            
            # Close all active sessions
            for session in list(self._active_sessions):
                try:
                    if session.in_transaction():
                        await session.rollback()
                    await session.close()
                except Exception as e:
                    bt.logging.error(f"Error closing session during cleanup: {e}")
            self._active_sessions.clear()
            
            # Final WAL cleanup and engine disposal
            try:
                async with self.get_session() as session:
                    await session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            except Exception as e:
                bt.logging.error(f"Error during WAL cleanup: {e}")
            
            try:
                await self.engine.dispose()
            except Exception as e:
                bt.logging.error(f"Error disposing engine: {e}")
            
        except Exception as e:
            bt.logging.error(f"Error during database cleanup: {e}")
        finally:
            self._shutting_down = False

    async def _cleanup_sessions(self):
        """Background task to cleanup stale sessions and connections"""
        while not self._cleanup_event.is_set():
            try:
                await self._cleanup_stale_connections()
            except Exception as e:
                bt.logging.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)  # Run cleanup every minute

    async def _safe_close_session(self, session):
        """Safely close a session with timeout"""
        if session:
            try:
                async with async_timeout.timeout(1):
                    await session.close()
            except Exception as e:
                bt.logging.debug(f"Error closing session: {e}")

    @asynccontextmanager
    async def get_session(self):
        """Session context manager with improved error handling and connection management"""
        if self._shutting_down:
            raise RuntimeError("Database manager is shutting down")
            
        session = None
        connection = None
        try:
            # Try to acquire lock with timeout
      
            # Cleanup stale connections first
            await self._cleanup_stale_connections()
            
            # Get connection first
            connection = await self._acquire_connection()
            
            # Create session with acquired connection
            session = self.async_session(bind=connection)
            session.created_at = time.time()
            self._active_sessions.add(session)
            
            # Set pragmas for this session with shorter timeout
            async with async_timeout.timeout(5):
                await session.execute(text("PRAGMA journal_mode = WAL"))
                await session.execute(text("PRAGMA synchronous = NORMAL"))
                await session.execute(text("PRAGMA busy_timeout = 30000"))
                await session.execute(text("PRAGMA temp_store = MEMORY"))
                await session.execute(text("PRAGMA cache_size = -2000"))
                
            yield session
                
        except asyncio.CancelledError:
            bt.logging.warning("Session operation cancelled, performing cleanup")
            await self._safe_close_session(session)
            if connection:
                await self._safe_close_connection(connection)
            raise
        except SQLAlchemyError as e:
            bt.logging.error(f"Database error: {e}")
            await self._safe_close_session(session)
            if connection:
                await self._safe_close_connection(connection)
            raise
        except asyncio.TimeoutError:
            bt.logging.error("Timeout waiting for database connection lock")
            await self._safe_close_session(session)
            if connection:
                await self._safe_close_connection(connection)
            raise
        except Exception as e:
            bt.logging.error(f"Unexpected error in get_session: {e}")
            await self._safe_close_session(session)
            if connection:
                await self._safe_close_connection(connection)
            raise
        finally:
            await self._safe_close_session(session)
            if connection:
                await self._safe_close_connection(connection)

    @asynccontextmanager
    async def get_long_running_session(self):
        """Session context manager optimized for long-running operations"""
        if self._shutting_down:
            raise RuntimeError("Database manager is shutting down")
        
        session = None
        connection = None
        try:
            # Get dedicated connection for long-running operation
            connection = await self._acquire_connection()
            
            # Create session with acquired connection
            session = self.async_session(bind=connection)
            session.created_at = time.time()
            session.is_long_running = True  # Mark as long-running
            self._active_sessions.add(session)
            
            # Set optimized pragmas for long-running operations
            await session.execute(text("PRAGMA journal_mode = WAL"))
            await session.execute(text("PRAGMA synchronous = NORMAL"))
            await session.execute(text("PRAGMA busy_timeout = 120000"))  # 2 minute timeout
            await session.execute(text("PRAGMA temp_store = MEMORY"))
            await session.execute(text("PRAGMA cache_size = -4000"))  # 4MB cache
            await session.execute(text("PRAGMA page_size = 4096"))
            await session.execute(text("PRAGMA mmap_size = 268435456"))  # 256MB mmap
            
            yield session
            
        except Exception as e:
            bt.logging.error(f"Error in long-running session: {e}")
            await self._safe_close_session(session)
            if connection:
                await self._safe_close_connection(connection)
            raise
        finally:
            await self._safe_close_session(session)
            if connection:
                await self._safe_close_connection(connection)

    def _convert_params(self, params):
        """Convert parameters to SQLAlchemy-compatible format"""
        if params is None:
            return {}
        elif isinstance(params, list):
            if not params:
                return {}
            if isinstance(params[0], (tuple, dict)):
                return params
            # This conversion could cause issues with hotkey/coldkey values
            return [{"param": p} for p in params]
        elif isinstance(params, tuple):
            # This conversion could replace column names with p0, p1, etc.
            return {f"p{i}": val for i, val in enumerate(params)}
        elif isinstance(params, dict):
            return params
        else:
            return {"param": params}

    async def execute_query(self, query, params=None):
        """Execute query with retry logic"""
        async with self.get_session() as session:
            if params is None:
                params = {}
            elif isinstance(params, list):
                # Convert list of values to list of dictionaries
                if params and not isinstance(params[0], (tuple, dict)):
                    params = [{"param": p} for p in params]
                    query = query.replace("?", ":param")
            elif isinstance(params, (list, tuple)):
                # Convert single tuple to dict
                counter = itertools.count()
                params = {f"p{i}": val for i, val in enumerate(params)}
                query = re.sub(r'\?', lambda m: f":p{next(counter)}", query)
            
            cursor = await session.execute(text(query), params)
            await session.commit()
            
            if query.strip().upper().startswith('SELECT'):
                return cursor
            return None

    async def fetch_all(self, query, params=None):
        """Execute a SELECT query and return all results."""
        try:
            async with async_timeout.timeout(self.default_timeout):  # Use the class default_timeout
                async with self.get_session() as session:
                    result = await session.execute(text(query), params)
                    return [dict(row) for row in result.mappings()]
        except asyncio.TimeoutError:
            bt.logging.error(f"Query timed out after {self.default_timeout} seconds")
            raise
        except Exception as e:
            bt.logging.error(f"Error in fetch_all: {str(e)}")
            bt.logging.error(traceback.format_exc())
            raise

    async def fetch_one(self, query, params=None):
        """Fetch single record maintaining old format"""
        async with self.get_session() as session:
            # Handle SQLite-style positional parameters (?)
            if params and isinstance(params, (list, tuple)):
                counter = itertools.count()
                params = {f"p{i}": val for i, val in enumerate(params)}
                query = re.sub(r'\?', lambda m: f":p{next(counter)}", query)
            
            result = await session.execute(text(query), params or {})
            if not result.returns_rows:
                return None
            row = result.first()
            if row is None:
                return None
            return dict(zip(result.keys(), row))

    async def executemany(self, query, params_list, column_names=None, max_retries=5, retry_delay=1):
        """Execute many queries with proper column name handling"""
        if not params_list:
            return
        
        batch_size = 25
        for i in range(0, len(params_list), batch_size):
            batch = params_list[i:i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    async with self.get_session() as session:
                        if batch and isinstance(batch[0], (list, tuple)):
                            if column_names:
                                # Use provided column names
                                params_dicts = [
                                    {col: val for col, val in zip(column_names, params)}
                                    for params in batch
                                ]
                                # Replace ? with :column_name
                                query_converted = query
                                for col in column_names:
                                    query_converted = query_converted.replace('?', f":{col}", 1)
                            else:
                                # Fallback to generic names if no column names provided
                                param_count = query.count('?')
                                param_names = [f"p{i}" for i in range(param_count)]
                                params_dicts = [
                                    {f"p{i}": val for i, val in enumerate(params)}
                                    for params in batch
                                ]
                                query_converted = query
                                for name in param_names:
                                    query_converted = query_converted.replace('?', f":{name}", 1)
                        else:
                            params_dicts = batch
                            query_converted = query
                        
                        await session.execute(text(query_converted), params_dicts)
                        await session.commit()
                        break
                
                except SQLAlchemyError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        bt.logging.warning(
                            f"Database locked during batch operation, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise
                
            # Small delay between batches to reduce contention
            if i < len(params_list) - batch_size:
                await asyncio.sleep(0.1)

    async def begin_transaction(self):
        """Begin a new transaction."""
        if self._operations_paused:
            raise OperationalError("Database operations are paused")
        
        transaction_id = str(uuid.uuid4())
        
        try:
            if self._transaction_in_progress:
                raise sqlite3.OperationalError("Transaction already in progress")
                
            if not await self.ensure_connection():
                raise ConnectionError("No database connection")
            
            # Check transaction state using sqlite_master
            cursor = await self.conn.execute("SELECT COUNT(*) FROM sqlite_master")
            await cursor.fetchone()  # This will fail if we're in a failed transaction state
            
            await self.conn.execute("BEGIN IMMEDIATE")
            self._active_transactions.add(transaction_id)
            self._transaction_in_progress = True
            bt.logging.debug(f"Transaction {transaction_id} started")
            return transaction_id
                
        except Exception as e:
            bt.logging.error(f"Error starting transaction: {e}")
            if transaction_id in self._active_transactions:
                self._active_transactions.remove(transaction_id)
            self._transaction_in_progress = False
            raise

    async def commit_transaction(self, transaction_id):
        """Commit a specific transaction with timeout."""
        try:
            async with async_timeout.timeout(5):
            
                if transaction_id not in self._active_transactions:
                    bt.logging.warning(f"Transaction {transaction_id} not found")
                    return
                    
                await self.conn.commit()
                self._active_transactions.remove(transaction_id)
                self._transaction_in_progress = False
                bt.logging.debug(f"Transaction {transaction_id} committed")
                    
        except Exception as e:
            bt.logging.error(f"Error committing transaction: {e}")
            # Force cleanup on error
            if transaction_id in self._active_transactions:
                self._active_transactions.remove(transaction_id)
            self._transaction_in_progress = False
            raise
        finally:
            # Try checkpoint without transaction lock
            try:
                await self.checkpoint_if_needed()
            except Exception as e:
                bt.logging.warning(f"Post-commit checkpoint failed: {e}")

    async def rollback_transaction(self, transaction_id):
        """Rollback a specific transaction."""
        try:
    
            if transaction_id not in self._active_transactions:
                return
                
            async with self._lock:
                await self.conn.rollback()
                self._active_transactions.remove(transaction_id)
                if not self._active_transactions:
                    self._transaction_in_progress = False
                bt.logging.debug(f"Transaction {transaction_id} rolled back")
                
        except Exception as e:
            bt.logging.error(f"Error rolling back transaction: {e}")
            raise

    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def cleanup(self):
        """Clean shutdown of database connections"""
        try:
            async with self.get_session() as session:
                # Try progressive checkpointing
                try:
                    await session.execute(text("PRAGMA wal_checkpoint(PASSIVE)"))
                    await asyncio.sleep(1)
                    
                    await session.execute(text("PRAGMA wal_checkpoint(RESTART)"))
                    await asyncio.sleep(1)
                    
                    await session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                except Exception as e:
                    bt.logging.error(f"Checkpoint error during cleanup: {e}")
                
                # Ensure WAL mode is disabled before closing
                await session.execute(text("PRAGMA journal_mode = DELETE"))
                
                # Close all connections
                await self.engine.dispose()
                
                # Verify WAL cleanup
                wal_path = Path(self.db_path).with_suffix('.db-wal')
                if wal_path.exists():
                    size = wal_path.stat().st_size
                    if size > 0:
                        bt.logging.warning(f"WAL file still has {size} bytes after cleanup")
                    else:
                        bt.logging.info("WAL file successfully cleared")
                    
        except Exception as e:
            bt.logging.error(f"Error during database cleanup: {e}")
            raise

    
    
    async def reconnect(self):
        """Force a reconnection to the database"""
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

    async def wait_for_locks_to_clear(self, timeout=30):
        """Wait for any database locks to clear with timeout and progressive checkpointing"""
        start_time = time.time()
        last_size = None
        last_checkpoint_time = 0
        CHECKPOINT_INTERVAL = 5  # Try checkpoint every 5 seconds
        
        try:
            async with async_timeout.timeout(timeout):
                while True:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Check WAL file size
                    wal_path = Path(self.db_path).with_suffix('.db-wal')
                    current_size = wal_path.stat().st_size if wal_path.exists() else 0
                    
                    # Log progress
                    if last_size is not None:
                        if current_size < last_size:
                            bt.logging.debug(f"WAL file shrinking: {last_size} -> {current_size}")
                        elif current_size > last_size:
                            bt.logging.debug(f"WAL file growing: {last_size} -> {current_size}")
                    
                    # Try progressive checkpointing
                    if current_time - last_checkpoint_time > CHECKPOINT_INTERVAL:
                        try:
                            if current_size > 10_000_000:  # 10MB
                                await self._safe_checkpoint("PASSIVE")
                            elif current_size > 1_000_000:  # 1MB
                                await self._safe_checkpoint("RESTART")
                            else:
                                await self._safe_checkpoint("TRUNCATE")
                            last_checkpoint_time = current_time
                        except Exception as e:
                            bt.logging.warning(f"Checkpoint attempt failed: {e}")
                    
                    # Exit conditions
                    if current_size == 0:
                        return True
                    
                    if elapsed > timeout * 0.8:  # If we're near timeout
                        bt.logging.warning(f"Timeout approaching - final WAL size: {current_size}")
                        return False
                    
                    last_size = current_size
                    await asyncio.sleep(0.5)
                    
        except asyncio.TimeoutError:
            bt.logging.error(f"Timeout waiting for locks to clear (Final WAL size: {current_size})")
            raise
        except asyncio.CancelledError:
            bt.logging.warning("Lock clearing operation cancelled")
            raise
        except Exception as e:
            bt.logging.error(f"Error while waiting for locks: {e}")
            raise

    async def _safe_checkpoint(self, checkpoint_type="TRUNCATE"):
        """Execute checkpoint with proper error handling"""
        try:
            async with self.get_session() as session:
                await session.execute(text(f"PRAGMA wal_checkpoint({checkpoint_type})"))
        except Exception as e:
            bt.logging.debug(f"Checkpoint ({checkpoint_type}) failed: {e}")
            raise

    async def checkpoint_if_needed(self, force=False):
        """Improved checkpoint handling"""
        try:
            wal_path = Path(self.db_path).with_suffix('.db-wal')
            if force or (wal_path.exists() and wal_path.stat().st_size > 1024 * 1024):  # 1MB threshold
                async with self.get_session() as session:
                    # Try progressive checkpointing
                    for mode in ['PASSIVE', 'RESTART', 'TRUNCATE']:
                        try:
                            await session.execute(text(f"PRAGMA wal_checkpoint({mode})"))
                            await asyncio.sleep(1)
                        except Exception as e:
                            bt.logging.warning(f"{mode} checkpoint failed: {e}")
                            continue
                        
                    # Verify checkpoint success
                    if wal_path.exists():
                        size = wal_path.stat().st_size
                        if size > 0:
                            bt.logging.warning(f"WAL file still has {size} bytes after checkpoint")
                        else:
                            bt.logging.debug("WAL file successfully cleared")
                            
        except Exception as e:
            bt.logging.error(f"Checkpoint failed: {e}")

    async def safe_shutdown(self):
        """Safely shutdown database and clear WAL file"""
        try:
            async with self.get_session() as session:
                # First try PASSIVE checkpoint
                await session.execute(text("PRAGMA wal_checkpoint(PASSIVE)"))
                await asyncio.sleep(1)
                
                # Then RESTART checkpoint
                await session.execute(text("PRAGMA wal_checkpoint(RESTART)"))
                await asyncio.sleep(1)
                
                # Finally TRUNCATE checkpoint
                await session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                
                # Verify WAL is cleared
                wal_path = Path(self.db_path).with_suffix('.db-wal')
                if wal_path.exists():
                    size = wal_path.stat().st_size
                    if size > 0:
                        bt.logging.warning(f"WAL file still has {size} bytes after checkpoint")
                    else:
                        bt.logging.info("WAL file successfully cleared")
                    
        except Exception as e:
            bt.logging.error(f"Error during safe shutdown: {e}")
            raise
        finally:
            await self.engine.dispose()

    async def create_backup_session(self):
        """Create a dedicated backup session with specific settings"""
        backup_engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            poolclass=AsyncAdaptedQueuePool,
            pool_size=1,  # Dedicated connection
            max_overflow=0,
            isolation_level='SERIALIZABLE',  # Ensure consistency
            echo=False
        )
        
        backup_session = sessionmaker(
            backup_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        return backup_session

    async def prepare_for_backup(self):
        """Prepare database for backup by ensuring WAL is checkpointed"""
        try:
            async with self.get_session() as session:
                # Try progressive checkpointing
                for checkpoint_mode in ['PASSIVE', 'RESTART', 'TRUNCATE']:
                    try:
                        await session.execute(text(f"PRAGMA wal_checkpoint({checkpoint_mode})"))
                        await session.commit()
                        await asyncio.sleep(1)  # Give other operations time to complete
                    except SQLAlchemyError as e:
                        bt.logging.warning(f"Checkpoint {checkpoint_mode} failed: {e}")
                        continue
                
                # Verify WAL size
                result = await session.execute(text("PRAGMA wal_size"))
                wal_size = (await result.scalar()) or 0
                
                if wal_size > 0:
                    bt.logging.warning(f"WAL still has {wal_size} bytes after checkpoint")
                    
                return wal_size == 0
                
        except Exception as e:
            bt.logging.error(f"Error preparing for backup: {e}")
            return False

    async def create_backup(self, backup_path: Path) -> bool:
        """
        Create a backup of the current database using SQLite's backup API.

        Args:
            backup_path (Path): The path where the backup will be stored.

        Returns:
            bool: True if backup is successful, False otherwise.
        """
        try:
            # Use separate connections for source and destination
            async with aiosqlite.connect(self.database_path) as source_conn:
                async with aiosqlite.connect(str(backup_path)) as dest_conn:
                    await source_conn.backup(dest_conn)
            bt.logging.info(f"Database backup created at {backup_path}")
            return True
        except Exception as e:
            bt.logging.error(f"Failed to create database backup: {e}")
            return False

    async def verify_backup(self, backup_path: Path) -> bool:
        """
        Verify the integrity of the backup database.

        Args:
            backup_path (Path): The path to the backup database.

        Returns:
            bool: True if backup is valid, False otherwise.
        """
        try:
            async with aiosqlite.connect(str(backup_path)) as conn:
                async with conn.execute("PRAGMA integrity_check;") as cursor:
                    result = await cursor.fetchone()
                    if result[0].lower() == "ok":
                        bt.logging.info("Backup integrity check passed.")
                        return True
                    else:
                        bt.logging.error(f"Backup integrity check failed: {result[0]}")
                        return False
        except Exception as e:
            bt.logging.error(f"Failed to verify backup: {e}")
            return False

    async def create_verified_backup(self, backup_path: Path) -> bool:
        """Create and verify a backup using SQLAlchemy"""
        try:
            await self.create_backup(backup_path)
            is_valid = await self.verify_backup(backup_path)
            return is_valid
        except Exception as e:
            bt.logging.error(f"Backup creation or verification failed: {e}")
            return False

    async def dispose(self):
        """Dispose the engine properly."""
        await self.engine.dispose()

    async def update_miner_weights(self, weight_updates, max_retries=5, retry_delay=1):
        """Specialized method for updating miner weights with optimized batching."""
        if not weight_updates:
            return

        # Sort updates by miner_uid to reduce lock contention
        weight_updates.sort(key=lambda x: x[1])
        
        # Break updates into smaller batches
        batch_size = 25
        total_batches = (len(weight_updates) + batch_size - 1) // batch_size
        
        query = """
            UPDATE miner_stats 
            SET most_recent_weight = :weight 
            WHERE miner_uid = :miner_uid
        """
        
        for batch_num, i in enumerate(range(0, len(weight_updates), batch_size)):
            batch = weight_updates[i:i + batch_size]
            bt.logging.debug(f"Processing weight update batch {batch_num + 1}/{total_batches} ({len(batch)} miners)")
            
            for attempt in range(max_retries):
                try:
                    async with self.get_session() as session:
                        # Set pragmas for better concurrency
                        await session.execute(text("PRAGMA journal_mode = WAL"))
                        await session.execute(text("PRAGMA synchronous = NORMAL"))
                        await session.execute(text("PRAGMA busy_timeout = 60000"))
                        await session.execute(text("PRAGMA temp_store = MEMORY"))
                        
                        # Convert tuples to dicts for SQLAlchemy
                        params_dicts = [
                            {"weight": float(weight), "miner_uid": int(uid)}
                            for weight, uid in batch
                        ]
                        
                        # Execute and commit in one go
                        await session.execute(text(query), params_dicts)
                        await session.commit()
                        
                        # Checkpoint if WAL file is getting large
                        await self.checkpoint_if_needed()
                        break  # Success - exit retry loop
                        
                except SQLAlchemyError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        bt.logging.warning(
                            f"Database locked during weight update batch {batch_num + 1}, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        
                        # Try to checkpoint and clean up WAL file
                        try:
                            async with self.get_session() as session:
                                await session.execute(text("PRAGMA wal_checkpoint(PASSIVE)"))
                        except:
                            pass
                            
                        continue
                    raise
                
            # Small delay between batches to reduce contention
            if batch_num < total_batches - 1:
                await asyncio.sleep(0.1)

    async def initialize(self):
        """Initialize database with optimized settings"""
        try:
            async with self.get_session() as session:
                # Set optimized pragmas
                await session.execute(text("PRAGMA journal_mode = WAL"))
                await session.execute(text("PRAGMA synchronous = NORMAL"))
                await session.execute(text("PRAGMA busy_timeout = 60000"))
                await session.execute(text("PRAGMA temp_store = MEMORY"))
                await session.execute(text("PRAGMA cache_size = -2000"))
                await session.execute(text("PRAGMA page_size = 4096"))
                await session.execute(text("PRAGMA mmap_size = 268435456"))
                await session.execute(text("PRAGMA wal_autocheckpoint = 1000"))
                await session.commit()
                
                bt.logging.info("Database initialized with optimized settings")
        except Exception as e:
            bt.logging.error(f"Error initializing database: {e}")
            raise

    @asynccontextmanager
    async def transaction(self):
        """Context manager for explicit transaction management with proper cleanup"""
        session = None
        try:
            session = self.async_session()
            async with session.begin():
                yield session
        except asyncio.CancelledError:
            bt.logging.warning("Transaction cancelled, performing cleanup")
            if session and session.in_transaction():
                try:
                    await session.rollback()
                except Exception as e:
                    bt.logging.error(f"Error rolling back cancelled transaction: {e}")
            raise
        except Exception as e:
            bt.logging.error(f"Error in transaction: {e}")
            if session and session.in_transaction():
                try:
                    await session.rollback()
                except Exception as rollback_error:
                    bt.logging.error(f"Error rolling back failed transaction: {rollback_error}")
            raise
        finally:
            if session:
                try:
                    await session.close()
                except Exception as e:
                    bt.logging.error(f"Error closing transaction session: {e}")

    async def ensure_connection(self):
        """Ensure database connection is active and valid"""
        try:
            # Test current engine
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            bt.logging.warning(f"Connection test failed: {e}")
            
            try:
                # Dispose old engine
                await self.engine.dispose()
                
                # Create new engine with optimized settings
                self.engine = create_async_engine(
                    f"sqlite+aiosqlite:///{self.database_path}",
                    echo=False,
                    pool_size=10,
                    max_overflow=20,
                    pool_timeout=30,
                    pool_recycle=1800,
                    connect_args={
                        "timeout": 60,
                        "check_same_thread": False,
                        "isolation_level": None,
                    }
                )
                
                # Test new engine
                async with self.engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                    return True
                    
            except Exception as reinit_error:
                bt.logging.error(f"Failed to reinitialize connection: {reinit_error}")
                return False

    async def _test_connection(self):
        """Test if the database connection is working"""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            bt.logging.error(f"Connection test failed: {e}")
            return False


