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
import ssdeep  # You'll need to pip install ssdeep
import sqlite3

class StateSync:
    def __init__(self, 
                 repo_url: str,
                 repo_branch: str,
                 state_dir: str = "./state"):
        
        # Store both HTTPS and SSH URLs
        self.https_url = repo_url if repo_url.startswith('https://') else f'https://github.com/{repo_url.split(":")[-1]}'
        
        # SSH URL only needed for pushing (primary node)
        if repo_url.startswith('https://github.com/'):
            repo_path = repo_url.replace('https://github.com/', '')
            self.ssh_url = f'git@github.com-datasync:{repo_path}'
        else:
            self.ssh_url = repo_url
            
        # Use HTTPS by default (for pulling)
        self.repo_url = self.https_url
        
        # Use the non-password protected deploy key
        self.ssh_key_path = os.path.expanduser('~/.ssh/data_sync_deploy')
        
        # Set up git SSH command to use specific key
        os.environ['GIT_SSH_COMMAND'] = f'ssh -i {self.ssh_key_path} -o IdentitiesOnly=yes'
        
        # Validate branch name
        if repo_branch not in ["main", "test"]:
            bt.logging.warning(f"Invalid branch. Setting to main by default for data sync")
            repo_branch = "main"
        self.repo_branch = repo_branch
        
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Files to sync
        self.state_files = [
            "validator.db",
            "state.pt",
            "entropy_system_state.json"
        ]

        # Files to track with Git LFS
        self.lfs_files = [
            "*.pt",      # PyTorch state files
            "*.db",      # Database files
            "*.json"     # Large JSON files
        ]

        # Add hash file to state files
        self.state_files.append("state_hashes.txt")
        self.hash_file = self.state_dir / "state_hashes.txt"

        # Add metadata tracking
        self.metadata_file = self.state_dir / "state_metadata.json"

        bt.logging.debug(f"Initialized StateSync with repo URL: {self.repo_url}")

    def _ensure_git_lfs(self, env):
        """Ensure Git LFS is installed and configured"""
        try:
            # Check if git-lfs is installed
            result = subprocess.run(
                ["git", "lfs", "version"],
                capture_output=True,
                env=env
            )
            if result.returncode != 0:
                bt.logging.error("Git LFS not installed. Please install git-lfs")
                return False
            
            # Initialize LFS in the repository
            subprocess.run(
                ["git", "lfs", "install"],
                cwd=self.state_dir,
                check=True,
                env=env
            )
            
            # Track files with LFS
            for pattern in self.lfs_files:
                subprocess.run(
                    ["git", "lfs", "track", pattern],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
            
            # Add .gitattributes to git
            subprocess.run(
                ["git", "add", ".gitattributes"],
                cwd=self.state_dir,
                check=True,
                env=env
            )
            
            return True
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Git LFS setup failed: {e}")
            return False

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
        try:
            # Pull latest metadata from remote
            subprocess.run(
                ["git", "fetch", "origin", self.repo_branch],
                cwd=self.state_dir,
                check=True
            )
            
            # Get remote metadata
            result = subprocess.run(
                ["git", "show", f"origin/{self.repo_branch}:state_metadata.json"],
                cwd=self.state_dir,
                capture_output=True,
                text=True
            )
            remote_metadata = json.loads(result.stdout)
            
            # Load local metadata
            if not self.metadata_file.exists():
                return True
                
            with open(self.metadata_file) as f:
                local_metadata = json.load(f)
            
            remote_update = datetime.fromisoformat(remote_metadata["last_update"])
            local_update = datetime.fromisoformat(local_metadata["last_update"])
            
            # If remote is older than local, don't pull
            if remote_update < local_update:
                return False
                
            # If remote is more than 20 minutes newer than local, pull
            if (remote_update - local_update) > timedelta(minutes=20):
                return True
            
            # Special handling for validator.db
            if "validator.db" in remote_metadata["files"]:
                remote_db_meta = remote_metadata["files"]["validator.db"]
                local_db_meta = local_metadata["files"].get("validator.db", {})
                
                if self._compare_db_states(local_db_meta, remote_db_meta):
                    return True
            
            # Compare other files using fuzzy hashing
            similarities = []
            for file, remote_data in remote_metadata["files"].items():
                if file in local_metadata["files"]:
                    local_data = local_metadata["files"][file]
                    
                    # Compare other files using fuzzy hashing
                    similarity = self._compare_fuzzy_hashes(
                        remote_data["hash"],
                        local_data["hash"]
                    )
                    similarities.append(similarity)
            
            # If average similarity is less than 80%, pull state
            if similarities and (sum(similarities) / len(similarities)) < 80:
                return True
                
            return False
            
        except Exception as e:
            bt.logging.error(f"Error checking state status: {e}")
            return False

    def push_state(self):
        """Push state using SSH (only for primary node)"""
        # Update metadata before pushing
        if not self._update_metadata_file():
            return False
            
        # Update hash file before pushing
        if not self._update_hash_file():
            return False
            
        try:
            env = os.environ.copy()
            
            # Force use of deploy key
            ssh_command = f'ssh -v -i {self.ssh_key_path} -o IdentitiesOnly=yes -o UserKnownHostsFile=/root/.ssh/known_hosts'
            env['GIT_SSH_COMMAND'] = ssh_command
            
            # Initialize repository if needed
            if not (self.state_dir / ".git").exists():
                bt.logging.debug("Initializing new git repository")
                subprocess.run(["git", "init"], cwd=self.state_dir, check=True, env=env)
                
                subprocess.run(
                    ["git", "remote", "add", "origin", self.ssh_url],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
                
                # Set up Git LFS
                if not self._ensure_git_lfs(env):
                    return False
            else:
                # Update remote URL to SSH
                subprocess.run(
                    ["git", "remote", "set-url", "origin", self.ssh_url],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
            
            # Create or switch to branch
            try:
                subprocess.run(
                    ["git", "checkout", self.repo_branch],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
            except subprocess.CalledProcessError:
                # Branch doesn't exist locally, create it
                subprocess.run(
                    ["git", "checkout", "-b", self.repo_branch],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
            
            # Add and commit changes
            subprocess.run(
                ["git", "add", "-f"] + self.state_files,
                cwd=self.state_dir,
                check=True,
                env=env
            )
            
            try:
                subprocess.run(
                    ["git", "commit", "-m", f"State update {datetime.now().isoformat()}"],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
            except subprocess.CalledProcessError:
                bt.logging.info("No changes to commit")
                return True
            
            # Push changes
            bt.logging.debug("Pushing changes using SSH")
            try:
                # Push LFS objects first
                subprocess.run(
                    ["git", "lfs", "push", "origin", self.repo_branch],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
                
                # Then push git changes (force push to ensure our state is preserved)
                subprocess.run(
                    ["git", "push", "-f", "origin", self.repo_branch],
                    cwd=self.state_dir,
                    check=True,
                    env=env
                )
                
                bt.logging.info(f"Successfully pushed state files to branch: {self.repo_branch}")
                return True
                
            except subprocess.CalledProcessError as e:
                bt.logging.error(f"Push failed: {e.stderr.decode() if e.stderr else str(e)}")
                return False
            
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Git command failed: {e.cmd} with return code {e.returncode}")
            bt.logging.error(f"Output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No output'}")
            return False

    def pull_state(self):
        """Pull latest state from GitHub for non-primary nodes using HTTPS"""
        try:
            bt.logging.debug(f"Pulling latest state from branch: {self.repo_branch}")
            
            # Initialize repository if needed
            if not (self.state_dir / ".git").exists():
                bt.logging.debug("Initializing new git repository")
                subprocess.run(["git", "init"], cwd=self.state_dir, check=True)
                subprocess.run(
                    ["git", "remote", "add", "origin", self.https_url],
                    cwd=self.state_dir,
                    check=True
                )
                
                # Set up Git LFS
                if not self._ensure_git_lfs({}):  # Empty env dict since we don't need SSH
                    return False
            else:
                # Update remote URL to HTTPS
                subprocess.run(
                    ["git", "remote", "set-url", "origin", self.https_url],
                    cwd=self.state_dir,
                    check=True
                )
            
            # Fetch all branches
            bt.logging.debug("Fetching all branches")
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=self.state_dir,
                check=True
            )
            
            # Reset local branch to match remote
            try:
                bt.logging.debug(f"Resetting to origin/{self.repo_branch}")
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{self.repo_branch}"],
                    cwd=self.state_dir,
                    check=True
                )
            except subprocess.CalledProcessError:
                # If branch doesn't exist locally, create it
                bt.logging.debug(f"Creating new branch: {self.repo_branch}")
                subprocess.run(
                    ["git", "checkout", "-b", self.repo_branch],
                    cwd=self.state_dir,
                    check=True
                )
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{self.repo_branch}"],
                    cwd=self.state_dir,
                    check=True
                )
            
            # Pull LFS files
            bt.logging.debug("Pulling LFS files")
            subprocess.run(
                ["git", "lfs", "pull"],
                cwd=self.state_dir,
                check=True
            )
            
            bt.logging.info(f"Successfully pulled latest state from branch: {self.repo_branch}")
            
            # After successful pull, check similarity
            if not self.check_state_similarity():
                bt.logging.warning("State files have diverged significantly. Consider re-syncing.")
            
            return True
            
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Git command failed: {e.cmd} with return code {e.returncode}")
            bt.logging.error(f"Output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No output'}")
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