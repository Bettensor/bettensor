import threading
import os
import subprocess
import json
import bittensor as bt

class Watchdog:
    """
    Handles random hanging issues by restarting the PM2 process.
    """
    def __init__(self, timeout):
        self.timeout = timeout
        self.timer = None
        self.pm2_process_name = self.get_pm2_process_name()

    def get_pm2_process_name(self):
        try:
            # Fetch the list of PM2 processes in JSON format
            result = subprocess.run(['pm2', 'jlist'], capture_output=True, text=True)
            processes = json.loads(result.stdout)
            current_pid = os.getpid()

            # Search for the process that matches the current PID
            for process in processes:
                if process['pid'] == current_pid:
                    return process['name']
            bt.logging.error("Failed to find the PM2 process name.")
            return None
        except Exception as e:
            bt.logging.error(f"Error determining PM2 process name: {str(e)}")
            return None

    def reset(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.handle_timeout)
        self.timer.start()

    def handle_timeout(self):
        bt.logging.error("Watchdog timer expired. Restarting process...")
        self.restart_process()

    def restart_process(self):
        if not self.pm2_process_name:
            bt.logging.error("PM2 process name is not set. Cannot restart the process.")
            return

        try:
            # Restart the PM2 process by name
            bt.logging.info(f"Attempting to restart PM2 process: {self.pm2_process_name}")
            subprocess.run(['pm2', 'restart', self.pm2_process_name], check=True)
            bt.logging.info(f"PM2 process {self.pm2_process_name} restart initiated")

            # Exit current process to prevent duplication
            os._exit(0)
        except Exception as e:
            bt.logging.error(f"Failed to restart PM2 process: {str(e)}")
            os._exit(1)

