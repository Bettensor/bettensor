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
        self.timer = threading.Timer(self.timeout, self.handle_timeout)
        self.timer.start()
        self.pm2_process_name = self.get_pm2_process_name()

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
