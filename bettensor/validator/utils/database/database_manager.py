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
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            poolclass=AsyncAdaptedQueuePool,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False
        )
        self.async_session = sessionmaker(
            self.engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
        self._transaction_in_progress = False
        self._lock = asyncio.Lock()
        self._connection_lock = asyncio.Lock()
        self._pause_lock = asyncio.Lock()
        self._operations_paused = False
        self._transaction_timeout = 120  # 30 second timeout
        self._transaction_lock = asyncio.Lock()
        self._active_transactions = set()

    @asynccontextmanager
    async def get_session(self):
        """Session context manager with error handling"""
        async with self.async_session() as session:
            try:
                yield session
            except SQLAlchemyError as e:
                await session.rollback()
                bt.logging.error(f"Database error: {e}")
                raise
            finally:
                await session.close()

    async def execute_query(self, query, params=None, max_retries=3):
        """Execute query with retry logic"""
        for attempt in range(max_retries):
            try:
                async with self.get_session() as session:
                    async with session.begin():
                        if params and isinstance(params, (list, tuple)):
                            counter = itertools.count()
                            params = {f"p{i}": val for i, val in enumerate(params)}
                            query = re.sub(r'\?', lambda m: f":p{next(counter)}", query)
                        elif params is None:
                            params = {}
                        
                        cursor = await session.execute(text(query), params)
                        if query.strip().upper().startswith('SELECT'):
                            return cursor
                        return None
                    
            except SQLAlchemyError as e:
                if "no active connection" in str(e) and attempt < max_retries - 1:
                    bt.logging.warning(f"Database connection lost, retrying ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(1)
                    await self.ensure_connection()
                    continue
                raise

    async def fetch_all(self, query, params=None):
        """Fetch all results maintaining old format"""
        async with self.get_session() as session:
            # Handle SQLite-style positional parameters (?)
            if params and isinstance(params, (list, tuple)):
                counter = itertools.count()
                params = {f"p{i}": val for i, val in enumerate(params)}
                query = re.sub(r'\?', lambda m: f":p{next(counter)}", query)
                
            result = await session.execute(text(query), params or {})
            if not result.returns_rows:
                return []
            # Return list of dicts instead of tuples
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.all()]

    async def executemany(self, query, params_list):
        """Execute many queries maintaining old format"""
        if not params_list:
            return
        
        async with self.get_session() as session:
            async with session.begin():
                # Handle SQLite-style positional parameters (?)
                if params_list and isinstance(params_list[0], (list, tuple)):
                    counter = itertools.count()
                    # Replace ? with :p0, :p1, etc. in query
                    param_count = query.count('?')
                    param_names = [f"p{i}" for i in range(param_count)]
                    query_converted = query
                    for i in range(param_count):
                        query_converted = query_converted.replace('?', f":p{i}", 1)
                    
                    # Convert list of tuples to list of dicts
                    params_dicts = []
                    for params in params_list:
                        param_dict = {f"p{i}": val for i, val in enumerate(params)}
                        params_dicts.append(param_dict)
                else:
                    params_dicts = params_list
                    query_converted = query
                
                await session.execute(text(query_converted), params_dicts)

    async def initialize(self, force=False):
        """Initialize database with optimized settings"""
        if self._initialized and not force:
            bt.logging.debug("Database already initialized and force is False")
            return

        async with self._instance_lock:
            self.engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.db_path}",
                poolclass=AsyncAdaptedQueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=False
            )

            async with self.engine.begin() as conn:
                # Optimized WAL settings
                pragmas = [
                    "PRAGMA journal_mode = WAL",
                    "PRAGMA synchronous = NORMAL",
                    "PRAGMA wal_autocheckpoint = 1000",
                    "PRAGMA busy_timeout = 5000",
                    "PRAGMA journal_size_limit = 32768",
                    "PRAGMA mmap_size = 30000000000",
                    "PRAGMA page_size = 4096",
                    "PRAGMA cache_size = -2000",
                    "PRAGMA temp_store = MEMORY"
                ]
                
                for pragma in pragmas:
                    await conn.execute(text(pragma))

                # Initialize schema
                statements = initialize_database()
                for statement in statements:
                    try:
                        await conn.execute(text(statement))
                        bt.logging.debug(f"Initialized database with statement: {statement}")
                    except SQLAlchemyError as e:
                        if "duplicate column" in str(e):
                            bt.logging.debug(f"Column exists: {e}")
                            continue
                        raise

            self._initialized = True

    async def ensure_connection(self):
        """Ensure database connection is active and valid"""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except sqlalchemy.exc.OperationalError:
            bt.logging.warning("Reinitializing database connection pool...")
            await self.engine.dispose()
            self.engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.db_path}",
                poolclass=AsyncAdaptedQueuePool,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                echo=False
            )
            self.async_session = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            return await self.ensure_connection()
        except Exception as e:
            bt.logging.error(f"Database connection error: {e}")
            return False

    async def reset_database_pragmas(self):
        """Reset database PRAGMAs to default values asynchronously."""
        try:
            # Set WAL mode with auto-checkpoint
            await self.conn.execute("PRAGMA journal_mode = WAL")
            await self.conn.execute("PRAGMA synchronous = NORMAL")
            await self.conn.execute("PRAGMA foreign_keys = ON")
            await self.conn.execute("PRAGMA cache_size = -2000")
            await self.conn.execute("PRAGMA wal_autocheckpoint = 100")  # Auto-checkpoint after 100 pages
            await self.conn.execute("PRAGMA busy_timeout = 5000")       # Wait up to 5 seconds for locks
            await self.conn.commit()
            bt.logging.debug("Database PRAGMAs reset")
        except Exception as e:
            bt.logging.error(f"Error resetting database PRAGMAs: {e}")
            raise

    async def execute_state_sync_query(self, query, params=None, max_retries=3, retry_delay=5):
        """Special query execution for state sync operations with retry logic"""
        query_start = time.time()
        query_type = query.strip().split()[0].upper()
        
        for attempt in range(max_retries):
            try:
                bt.logging.debug(f"Executing state sync query (attempt {attempt + 1}/{max_retries})")
                
                # Wait for any existing locks first
                await self.wait_for_locks_to_clear(timeout=retry_delay)
                
                async with self._lock:
                    async with self.get_session() as session:
                        async with session.begin():
                            await session.execute(text(query), params if params else {})
                            duration = time.time() - query_start
                            bt.logging.debug(f"State sync query completed in {duration:.3f}s")
                            
                            # Add checkpoint check after writes
                            if query_type != 'SELECT':
                                await self.checkpoint_if_needed()
                                
                            return True
                        
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        bt.logging.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        continue
                raise
            except Exception as e:
                bt.logging.error(f"State sync query error after {time.time() - query_start:.3f}s: {str(e)}")
                raise

    async def fetch_one(self, query, params=None):
        """Fetch single record maintaining old format"""
        async with self.get_session() as session:
            # Handle SQLite-style positional parameters (?)
            if params and isinstance(params, (list, tuple)):
                counter = itertools.count()
                # Convert to dict with positional names
                params = {f"p{i}": val for i, val in enumerate(params)}
                # Replace ? with :p0, :p1, etc.
                query = re.sub(r'\?', lambda m: f":p{next(counter)}", query)
                
            result = await session.execute(text(query), params or {})
            if not result.returns_rows:
                return None
            row = result.first()
            if row is None:
                return None
            columns = result.keys()
            return dict(zip(columns, row))

    async def begin_transaction(self):
        """Begin a new transaction with proper locking and timeout."""
        if self._operations_paused:
            raise OperationalError("Database operations are paused")
        
        transaction_id = str(uuid.uuid4())
        
        try:
            async with async_timeout.timeout(5):
                async with self._transaction_lock:
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
                    
        except asyncio.TimeoutError:
            bt.logging.error("Timeout waiting to begin transaction")
            raise
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
                async with self._transaction_lock:
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
            async with self._transaction_lock:
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
            async with aiosqlite.connect(self.db_path) as source_conn:
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


