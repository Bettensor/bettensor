import subprocess
import traceback
import bittensor as bt
import os
import json
import configparser

def get_current_branch():
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode().strip()
        return branch
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Failed to get current git branch: {e}")
        return None

def get_remote_hash(branch):
    try:
        remote_hash = subprocess.check_output(
            ["git", "rev-parse", f"origin/{branch}"]
        ).decode().strip()
        return remote_hash
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Failed to get remote hash for branch {branch}: {e}")
        return None

def get_local_hash():
    try:
        local_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
        return local_hash
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Failed to get local git hash: {e}")
        return None

def pull_latest_changes():
    try:
        current_branch = get_current_branch()
        if not current_branch:
            bt.logging.error("Failed to get current branch.")
            return False
        
        subprocess.check_call(["git", "fetch", "origin"])
        subprocess.check_call(["git", "pull", "origin", current_branch])
        
        bt.logging.info(f"Successfully pulled latest changes from {current_branch} branch.")
        return True
    except subprocess.CalledProcessError as e: 
        bt.logging.error(f"Failed to pull latest changes: {e}")
        return False

def get_pm2_process_name():
    try:
        # Get the PM2 process ID from the environment variable
        pm_id = os.environ.get('pm_id')
        
        if pm_id is None:
            return None  # Not running under PM2
        
        # Use PM2 to get process info
        result = subprocess.run(['pm2', 'jlist'], capture_output=True, text=True)
        processes = json.loads(result.stdout)
        
        # Find the process with matching pm_id
        for process in processes:
            if str(process.get('pm_id')) == pm_id:
                return process.get('name')
        
        return None  # Process not found
    except Exception as e:
        print(f"Error getting PM2 process name: {e}")
        return None

async def perform_update(validator):
    bt.logging.info("Update detected. Starting update procedure...")
    
    if pull_latest_changes():
        # Check if we need to reset scoring
        if check_version_and_reset():
            try:
                validator.reset_scoring_system()
                bt.logging.info("Scoring system reset successfully.")
                
                # Reset the flag in setup.cfg
                config = configparser.ConfigParser()
                config.read('setup.cfg')
                config.set('metadata', 'reset_validator_scores', 'False')
                with open('setup.cfg', 'w') as configfile:
                    config.write(configfile)
                
            except Exception as e:
                bt.logging.error(f"Failed to reset scoring system: {e}")
                bt.logging.error(traceback.format_exc())
        else:
            bt.logging.info("No scoring reset required for this update.")
            return
        
        # Restart the validator process managed by PM2
        try:
            process_name = get_pm2_process_name()
            if process_name:
                subprocess.check_call(["pm2", "restart", process_name,"--update-env"])
                bt.logging.info(f"Validator process '{process_name}' restarted successfully.")
            else:
                bt.logging.warning("Not running as a PM2 process. Manual restart may be required.")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Failed to restart validator process: {e}")
            bt.logging.error(traceback.format_exc())
    else:
        bt.logging.error("Update procedure aborted due to pull failure.")

def check_version_and_reset():
    config = configparser.ConfigParser()
    try:
        config.read('setup.cfg')
        reset_scores = config.getboolean('metadata', 'reset_validator_scores', fallback=False)
        return reset_scores
    except configparser.Error as e:
        bt.logging.error(f"Error reading setup.cfg: {e}")
        return False
