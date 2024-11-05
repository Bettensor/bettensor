import threading
import os
import subprocess
import json
import bittensor as bt
import time
import traceback


class Watchdog:
    """
    Handles random hanging issues by restarting the PM2 process.
    """

    def __init__(self, validator, timeout):
        self.validator = validator
        self.timeout = timeout
        self.last_reset = time.time()
        self.timer = threading.Timer(timeout, self._timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        self.pm2_process_name = self.get_pm2_process_name()

    def extend_timeout(self, additional_seconds):
        """Extends the current timeout by the specified number of seconds"""
        current_time = time.time()
        time_elapsed = current_time - self.last_reset
        remaining_time = self.timeout - time_elapsed
        
        # Only extend if we have less than 5 minutes remaining
        if remaining_time < 300:
            self.timer.cancel()
            new_timeout = remaining_time + additional_seconds
            self.timer = threading.Timer(new_timeout, self._timeout_handler)
            self.timer.daemon = True
            self.timer.start()
            bt.logging.debug(f"Extended watchdog timeout by {additional_seconds} seconds")

    def reset(self):
        self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.handle_timeout)
        self.timer.start()

    def handle_timeout(self):
        bt.logging.error("Watchdog timer expired. Restarting process...")
        self.restart_process()

    def restart_process(self):
        try:
            bt.logging.info(f"Attempting to restart PM2 process: {self.pm2_process_name}")
            subprocess.run(["pm2", "restart", self.pm2_process_name, "--update-env"], check=True)
            bt.logging.info(f"PM2 process {self.pm2_process_name} restarted successfully")
            bt.logging.info(f"PM2 process {self.pm2_process_name} restart initiated")

            # Exit current process to prevent duplication
            os._exit(0)
        except Exception as e:
            bt.logging.error(f"Failed to restart PM2 process: {str(e)}")
            os._exit(1)

    def get_pm2_process_name(self):
        try:
            result = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, check=True)
            processes = json.loads(result.stdout)
            for proc in processes:
                if proc.get("name") == "validator":
                    return proc.get("name")
            return None
        except Exception as e:
            bt.logging.error(f"Error fetching PM2 process list: {e}")
            return None

    def _timeout_handler(self):
        """Handle watchdog timeout by cleaning up and restarting"""
        bt.logging.error("Watchdog timer expired. Cleaning up resources...")
        
        try:
            # Get validator instance (assuming it's passed during initialization)
            if hasattr(self, 'validator'):
                # Cleanup database
                if hasattr(self.validator, 'db_manager'):
                    self.validator.db_manager.cleanup()
                
                # Cancel any pending async tasks
                if hasattr(self.validator, 'operation_lock'):
                    self.validator.operation_lock.release()
                
                # Reset any other resources
                if hasattr(self.validator, 'scoring_system'):
                    self.validator.scoring_system.save_state()
                    
        except Exception as e:
            bt.logging.error(f"Error during watchdog cleanup: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self.restart_process()
