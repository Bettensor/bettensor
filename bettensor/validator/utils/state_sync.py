import os
import sys
import time
import shutil
import requests
import subprocess
from datetime import datetime
import json
import logging
from pathlib import Path
import bittensor as bt

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

    def push_state(self):
        """Push state using SSH (only for primary node)"""
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
            return True
            
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Git command failed: {e.cmd} with return code {e.returncode}")
            bt.logging.error(f"Output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No output'}")
            return False