import os
import sys
import time
import shutil
import traceback
import requests
import subprocess
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import bittensor as bt
import ssdeep 
import sqlite3
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
from dotenv import load_dotenv
import stat  # Make sure to import stat module
import asyncio

class StateSync:
    def __init__(self, 
                 state_dir: str = "./state",
                 db_manager = None):
        """
        Initialize StateSync with Azure blob storage for both uploads and downloads
        
        Args:
            state_dir: Local directory for state files
            db_manager: Database manager instance
        """
        # Load environment variables
        load_dotenv()
        
        # Store database manager
        self.db_manager = db_manager
        
        #hardcoded read-only token for easy distribution - will issue tokens via api in the future
        readonly_token = "sp=r&st=2024-11-05T18:31:28Z&se=2039-11-06T02:31:28Z&spr=https&sv=2022-11-02&sr=c&sig=NJPxzJsi3zgVjHtJK5BNNYXUqxG5Hi0WfM4Fg3sgBB4%3D"

        # Get Azure configuration from environment
        sas_url = os.getenv('AZURE_STORAGE_SAS_URL','https://devbettensorstore.blob.core.windows.net')
        container_name = os.getenv('AZURE_STORAGE_CONTAINER','data')
        credential = os.getenv('VALIDATOR_API_TOKEN', readonly_token)

        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

       

        # Azure setup
        self.azure_enabled = bool(sas_url)
        if self.azure_enabled:
            self.blob_service = BlobServiceClient(account_url=sas_url, credential=credential)
            self.container = self.blob_service.get_container_client(container_name)
            bt.logging.info(f"Azure blob storage configured with container: {container_name}")
        else:
            bt.logging.error("Azure blob storage not configured - state sync will be disabled")

        self.state_files = [
            "validator.db",
            "state.pt", 
            "entropy_system_state.json",
            "state_hashes.txt",
            "state_metadata.json"
        ]

        self.hash_file = self.state_dir / "state_hashes.txt"
        self.metadata_file = self.state_dir / "state_metadata.json"


    def _compute_fuzzy_hash(self, filepath: Path) -> str:
        """Compute fuzzy hash of a file using ssdeep, with special handling for SQLite DB"""
        try:
            if filepath.name == "validator.db":
                # For SQLite DB, hash the table structure and row counts instead
                conn = sqlite3.connect(filepath)
                cursor = conn.cursor()
                
                # Get table schema and row counts
                hash_data = []
                
                # Get list of tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    # Get table schema
                    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    schema = cursor.fetchone()[0]
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    hash_data.append(f"{schema}:{count}")
                
                conn.close()
                
                # Create a stable string representation to hash
                db_state = "\n".join(sorted(hash_data))
                return ssdeep.hash(db_state.encode())
            else:
                # Regular file hashing
                with open(filepath, 'rb') as f:
                    return ssdeep.hash(f.read())
        except Exception as e:
            bt.logging.error(f"Error computing hash for {filepath}: {e}")
            return ""

    def _compare_fuzzy_hashes(self, hash1: str, hash2: str) -> int:
        """Compare two fuzzy hashes and return similarity percentage (0-100)"""
        try:
            if not hash1 or not hash2:
                return 0
            return ssdeep.compare(hash1, hash2)
        except Exception as e:
            bt.logging.error(f"Error comparing hashes: {e}")
            return 0

    def _update_hash_file(self):
        """Update the hash file with current file hashes"""
        try:
            hashes = {}
            for file in self.state_files:
                if file != "state_hashes.txt":  # Don't hash the hash file
                    filepath = self.state_dir / file
                    if filepath.exists():
                        hashes[file] = self._compute_fuzzy_hash(filepath)

            with open(self.hash_file, 'w') as f:
                json.dump(hashes, f, indent=2)
            return True
        except Exception as e:
            bt.logging.error(f"Error updating hash file: {e}")
            return False

    def check_state_similarity(self) -> bool:
        """Check if local files are sufficiently similar to stored hashes"""
        try:
            if not self.hash_file.exists():
                bt.logging.warning("No hash file found. Cannot compare states.")
                return False

            with open(self.hash_file, 'r') as f:
                stored_hashes = json.load(f)

            for file, stored_hash in stored_hashes.items():
                filepath = self.state_dir / file
                if filepath.exists():
                    current_hash = self._compute_fuzzy_hash(filepath)
                    similarity = self._compare_fuzzy_hashes(stored_hash, current_hash)
                    
                    if similarity < 80:  # Less than 80% similar
                        bt.logging.warning(f"File {file} has diverged significantly (similarity: {similarity}%)")
                        return False
                else:
                    bt.logging.warning(f"File {file} missing")
                    return False

            return True
        except Exception as e:
            bt.logging.error(f"Error checking state similarity: {e}")
            return False

    def _get_file_metadata(self, filepath: Path) -> dict:
        """Get metadata about a file including size and last modified time"""
        try:
            stats = filepath.stat()
            return {
                "size": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat(),
                "hash": self._compute_fuzzy_hash(filepath)
            }
        except Exception as e:
            bt.logging.error(f"Error getting metadata for {filepath}: {e}")
            return {}

    def _get_db_metadata(self, db_path: Path) -> dict:
        """Get detailed metadata about the database state"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            metadata = {
                "row_counts": {},
                "latest_dates": {},
                "size": db_path.stat().st_size,
                "modified": datetime.fromtimestamp(db_path.stat().st_mtime, timezone.utc).isoformat()
            }
            
            # Get row counts for key tables
            for table in ['predictions', 'game_data', 'miner_stats', 'scores']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                metadata["row_counts"][table] = cursor.fetchone()[0]
            
            # Get latest dates for time-sensitive tables
            cursor.execute("SELECT MAX(prediction_date) FROM predictions")
            metadata["latest_dates"]["predictions"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(event_start_date) FROM game_data")
            metadata["latest_dates"]["game_data"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(last_update_date) FROM score_state")
            metadata["latest_dates"]["score_state"] = cursor.fetchone()[0]
            
            conn.close()
            return metadata
            
        except Exception as e:
            bt.logging.error(f"Error getting database metadata: {e}")
            return {}

    def _compare_db_states(self, local_meta: dict, remote_meta: dict) -> bool:
        """
        Compare database states to determine if pull is needed
        Returns: True if state should be pulled, False otherwise
        """
        try:
            # If either metadata is empty, can't make a good comparison
            if not local_meta or not remote_meta:
                return False
                
            # Compare latest dates
            for table, remote_date in remote_meta["latest_dates"].items():
                local_date = local_meta["latest_dates"].get(table)
                
                if not local_date or not remote_date:
                    continue
                    
                remote_dt = datetime.fromisoformat(remote_date)
                local_dt = datetime.fromisoformat(local_date)
                
                # If remote has significantly newer data (>1 hour), pull
                if remote_dt > local_dt + timedelta(hours=1):
                    bt.logging.info(f"Remote has newer {table} data: {remote_dt} vs {local_dt}")
                    return True
                    
            # Compare row counts
            for table, remote_count in remote_meta["row_counts"].items():
                local_count = local_meta["row_counts"].get(table, 0)
                
                # If difference is more than 10%, pull
                if abs(remote_count - local_count) / max(remote_count, local_count) > 0.1:
                    bt.logging.info(f"Significant row count difference in {table}: {local_count} vs {remote_count}")
                    return True
            
            return False
            
        except Exception as e:
            bt.logging.error(f"Error comparing database states: {e}")
            return False

    def should_pull_state(self) -> bool:
        """
        Determine if state should be pulled based on similarity and timestamps
        Returns: True if state should be pulled, False otherwise
        """
        if not self.azure_enabled:
            bt.logging.error("Azure blob storage not configured")
            return False

        try:
            # Get remote metadata from Azure blob storage
            metadata_client = self.container.get_blob_client("state_metadata.json")
            temp_metadata = self.metadata_file.with_suffix('.tmp')
            
            try:
                # Download metadata file
                with open(temp_metadata, 'wb') as f:
                    stream = metadata_client.download_blob()
                    for chunk in stream.chunks():
                        f.write(chunk)
                
                # Load remote metadata
                with open(temp_metadata) as f:
                    remote_metadata = json.load(f)
                    
                return self._should_pull_state(remote_metadata)
                
            finally:
                # Clean up temp metadata file
                temp_metadata.unlink(missing_ok=True)
                
        except AzureError as e:
            bt.logging.error(f"Azure storage error checking state: {e}")
            return False
        except Exception as e:
            bt.logging.error(f"Error checking state status: {e}")
            return False

    async def pull_state(self):
        """Pull latest state from Azure blob storage if needed."""
        if not self.azure_enabled:
            return False

        try:
            # Get database manager instance
            db_manager = self.db_manager
            
            bt.logging.info("Starting state pull process...")
            
            # Debug: Check score_state before sync
            state_before = await db_manager.fetch_all(
                "SELECT state_id, current_day FROM score_state ORDER BY state_id DESC LIMIT 1"
            )
            bt.logging.debug(f"State before sync: {state_before}")
            
            # Wait for any pending writes to complete using WAL checkpoint
            bt.logging.info("Waiting for pending database operations to complete...")
            try:
                await db_manager.execute_query("PRAGMA wal_checkpoint(FULL)")
                # Additional check to ensure no active transactions
                active_queries = await db_manager.fetch_all("""
                    SELECT * FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'
                    AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='temp_table')
                """)
                while active_queries:
                    bt.logging.debug("Active transactions detected, waiting...")
                    await asyncio.sleep(1)
                    active_queries = await db_manager.fetch_all("""
                        SELECT * FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'
                        AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='temp_table')
                    """)
            except Exception as e:
                bt.logging.warning(f"Error checking WAL checkpoint: {e}")
            
            # Now pause database operations
            await db_manager.pause_operations()
            
            success = False
            try:
                # Create temporary directory for downloads
                temp_dir = self.state_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                
                try:
                    # Pull files from Azure to temp directory first
                    for filename in self.state_files:
                        blob_client = self.container.get_blob_client(filename)
                        temp_file = temp_dir / filename
                        
                        bt.logging.debug(f"Downloading {filename} to temporary location")
                        with open(temp_file, 'wb') as f:
                            stream = blob_client.download_blob()
                            for chunk in stream.chunks():
                                f.write(chunk)
                        
                        # Set initial permissions
                        os.chmod(temp_file, 0o666)
                    
                    # Special handling for database file
                    db_file = self.state_dir / "validator.db"
                    temp_db = temp_dir / "validator.db"
                    
                    if temp_db.exists():
                        bt.logging.info("Verifying downloaded database integrity...")
                        try:
                            # Test the downloaded database
                            test_conn = sqlite3.connect(temp_db)
                            test_conn.execute("PRAGMA integrity_check")
                            test_conn.close()
                            
                            # Close existing database connection
                            await db_manager.close()
                            
                            # Move the old database to .bak
                            if db_file.exists():
                                backup_file = db_file.with_suffix('.bak')
                                db_file.rename(backup_file)
                            
                            # Move new database into place
                            shutil.move(temp_db, db_file)
                            os.chmod(db_file, 0o666)
                            
                            # Move other state files
                            for filename in self.state_files:
                                if filename != "validator.db":
                                    src = temp_dir / filename
                                    dst = self.state_dir / filename
                                    if src.exists():
                                        shutil.move(src, dst)
                                        os.chmod(dst, 0o666)
                            
                            success = True
                            
                        except sqlite3.Error as e:
                            bt.logging.error(f"Database integrity check failed: {e}")
                            # Restore from backup if available
                            if db_file.with_suffix('.bak').exists():
                                shutil.move(db_file.with_suffix('.bak'), db_file)
                            raise
                    
                finally:
                    # Cleanup temp directory
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                
                if success:
                    # Reconnect to the new database
                    await db_manager.initialize()
                    
                    # Verify the new state
                    state_after = await db_manager.fetch_all(
                        "SELECT state_id, current_day FROM score_state ORDER BY state_id DESC LIMIT 1"
                    )
                    bt.logging.info(f"State after sync: {state_after}")
                    
                    # Update metadata and hashes
                    self._update_metadata_file()
                    self._update_hash_file()
                    
                    bt.logging.info("State pull completed successfully")
                
                return success
                
            except Exception as e:
                bt.logging.error(f"Error syncing files: {e}")
                bt.logging.error(traceback.format_exc())
                raise
            finally:
                # Resume database operations
                await db_manager.resume_operations()
                
        except Exception as e:
            bt.logging.error(f"Error pulling state: {str(e)}")
            bt.logging.error(traceback.format_exc())
            # Ensure operations are resumed even if there's an error
            if 'db_manager' in locals():
                await db_manager.resume_operations()
            return False

    def _should_pull_state(self, remote_metadata: dict) -> bool:
        """
        Determine if state should be pulled based on remote metadata
        """
        try:
            # Validate remote metadata structure
            required_fields = ["last_update", "files"]
            if not all(field in remote_metadata for field in required_fields):
                bt.logging.warning("Remote metadata missing required fields")
                return False
                
            # Load local metadata
            if not self.metadata_file.exists():
                bt.logging.info("No local metadata file - should pull state")
                return True
                
            try:
                with open(self.metadata_file) as f:
                    local_metadata = json.load(f)
            except json.JSONDecodeError:
                bt.logging.warning("Invalid local metadata file")
                return True
                
            # Validate local metadata structure
            if not all(field in local_metadata for field in required_fields):
                bt.logging.warning("Local metadata missing required fields")
                return True
            
            remote_update = datetime.fromisoformat(remote_metadata["last_update"])
            local_update = datetime.fromisoformat(local_metadata["last_update"])
            
            # If remote is older than local, don't pull
            if remote_update < local_update:
                bt.logging.debug("Remote state is older than local state")
                return False
                
            # If remote is more than 20 minutes newer than local, pull
            if (remote_update - local_update) > timedelta(minutes=20):
                bt.logging.info("Remote state is significantly newer")
                return True
            
            # Compare files
            for file, remote_data in remote_metadata["files"].items():
                if file not in local_metadata["files"]:
                    bt.logging.info(f"New file found in remote: {file}")
                    return True
                
                local_data = local_metadata["files"][file]
                
                # Validate hash exists in both metadata
                if "hash" not in remote_data or "hash" not in local_data:
                    bt.logging.debug(f"Missing hash for file {file}")
                    continue
                
                # Compare using fuzzy hashing
                similarity = self._compare_fuzzy_hashes(
                    remote_data["hash"],
                    local_data["hash"]
                )
                
                if similarity < 80:
                    bt.logging.info(f"File {file} has low similarity: {similarity}%")
                    return True
                
            return False
            
        except Exception as e:
            bt.logging.error(f"Error checking state status: {e}")
            return False

    def _update_metadata_file(self):
        """Update metadata file with current state information"""
        try:
            metadata = {
                "last_update": datetime.now(timezone.utc).isoformat(),
                "files": {}
            }
            
            for file in self.state_files:
                if file != "state_metadata.json":
                    filepath = self.state_dir / file
                    if filepath.exists():
                        if file == "validator.db":
                            metadata["files"][file] = self._get_db_metadata(filepath)
                        else:
                            metadata["files"][file] = self._get_file_metadata(filepath)

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            bt.logging.error(f"Error updating metadata: {e}")
            return False

    async def push_state(self):
        """Push state files to Azure blob storage"""
        if not self.azure_enabled:
            bt.logging.error("Azure blob storage not configured")
            return False

        try:
            # Get database manager instance
            db_manager = self.db_manager  # Use the instance we already have
            
            bt.logging.info("Preparing database for state push...")
            try:
                # Pause all ongoing operations while maintaining connection
                await db_manager.pause_operations()
                
                try:
                    # Force a checkpoint to ensure all changes are written to disk
                    bt.logging.debug("Executing WAL checkpoint...")
                    await db_manager.execute_state_sync_query("PRAGMA wal_checkpoint(FULL);")
                    
                    # Create a temporary copy of the database for pushing
                    db_path = self.state_dir / "validator.db"
                    temp_db = db_path.with_suffix('.pushing')
                    
                    bt.logging.debug(f"Creating temporary database copy: {temp_db}")
                    # Use shutil to create an atomic copy
                    shutil.copy2(db_path, temp_db)
                    
                    success = False
                    try:
                        # Upload files to Azure
                        for filename in self.state_files:
                            filepath = self.state_dir / filename
                            if not filepath.exists():
                                continue

                            # Use the temporary database file instead of the live one
                            if filename == "validator.db":
                                filepath = temp_db

                            blob_client = self.container.get_blob_client(filename)
                            with open(filepath, 'rb') as f:
                                bt.logging.debug(f"Uploading {filename} to Azure blob storage")
                                blob_client.upload_blob(f, overwrite=True)

                        success = True
                        
                    finally:
                        # Clean up temp file
                        if temp_db.exists():
                            temp_db.unlink()
                            bt.logging.debug("Temporary database copy removed")
                    
                    return success
                    
                finally:
                    # Always resume operations
                    await db_manager.resume_operations()
                    
            except Exception as e:
                bt.logging.error(f"Error in state sync: {str(e)}")
                bt.logging.error(traceback.format_exc())
                await db_manager.resume_operations()
                return False
                
        except Exception as e:
            bt.logging.error(f"Error pushing state: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False
