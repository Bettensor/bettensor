import os
import sys
import time
import shutil
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

class StateSync:
    def __init__(self, 
                 state_dir: str = "./state"):
        """
        Initialize StateSync with Azure blob storage for both uploads and downloads
        
        Args:
            state_dir: Local directory for state files
        """
        # Load environment variables
        load_dotenv()
        
        
        #hardcoded read-only token for easy distribution - will issue tokens via api in the futute
        readonly_token = "sp=r&st=2024-11-04T20:00:26Z&se=2024-11-05T04:00:26Z&spr=https&sv=2022-11-02&sr=c&sig=OIvDP%2FCSmRkGokddtGJLOGVDNbIf4YdvaH4BBb%2FZqQk%3D"

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
        """Compute fuzzy hash of a file using ssdeep"""
        try:
            with open(filepath, 'rb') as f:
                return ssdeep.hash(f.read())
        except Exception as e:
            bt.logging.error(f"Error computing hash for {filepath}: {e}")
            return ""

    def _compare_fuzzy_hashes(self, hash1: str, hash2: str) -> int:
        """Compare two fuzzy hashes and return similarity percentage (0-100)"""
        try:
            return ssdeep.compare(hash1, hash2)
        except Exception:
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

    def push_state(self):
        """Push state files to Azure blob storage (primary node only)"""
        if not self.azure_enabled:
            bt.logging.error("Azure blob storage not configured")
            return False

        if not self._update_metadata_file() or not self._update_hash_file():
            return False

        try:
            # Upload each file to Azure
            for filename in self.state_files:
                filepath = self.state_dir / filename
                if not filepath.exists():
                    continue

                # Get blob client for this file
                blob_client = self.container.get_blob_client(filename)
                
                # Upload with automatic chunking
                with open(filepath, 'rb') as f:
                    bt.logging.debug(f"Uploading {filename} to Azure blob storage")
                    blob_client.upload_blob(
                        f,
                        overwrite=True,
                        max_concurrency=4
                    )

            bt.logging.info("Successfully pushed state files to Azure")
            return True

        except AzureError as e:
            bt.logging.error(f"Azure storage error during push: {e}")
            return False
        except Exception as e:
            bt.logging.error(f"Error pushing state: {e}")
            return False

    def pull_state(self):
        """
        Pull latest state from Azure blob storage if needed.
        Returns: True if state was updated, False otherwise
        """
        if not self.azure_enabled:
            bt.logging.error("Azure blob storage not configured")
            return False

        try:
            # First pull and check metadata
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
                    
                # Check if we should pull
                should_update = self._should_pull_state(remote_metadata)
                
                if not should_update:
                    bt.logging.info("Local state is up to date")
                    temp_metadata.unlink()
                    return False
                    
                # If we should update, download all files
                for filename in self.state_files:
                    blob_client = self.container.get_blob_client(filename)
                    filepath = self.state_dir / filename
                    temp_file = filepath.with_suffix('.tmp')
                    
                    try:
                        with open(temp_file, 'wb') as f:
                            stream = blob_client.download_blob()
                            for chunk in stream.chunks():
                                f.write(chunk)
                        
                        # Atomic rename
                        temp_file.rename(filepath)
                        
                    except Exception as e:
                        temp_file.unlink(missing_ok=True)
                        raise e
                
                # If all files downloaded successfully, move metadata file into place
                temp_metadata.rename(self.metadata_file)
                
                bt.logging.info("Successfully pulled and updated state from Azure")
                return True
                
            finally:
                # Clean up temp metadata file if it exists
                temp_metadata.unlink(missing_ok=True)

        except AzureError as e:
            bt.logging.error(f"Azure storage error during pull: {e}")
            return False
        except Exception as e:
            bt.logging.error(f"Error pulling state: {e}")
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
