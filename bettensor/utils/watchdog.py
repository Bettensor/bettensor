import os
import signal
import sys
import subprocess

class Watchdog:
    """
    Handles random Hanging issue
    """
    def __init__(self, timeout):
        self.timeout = timeout
        self.timer = None

    def reset(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.handle_timeout)
        self.timer.start()

    def handle_timeout(self):
        bt.logging.error("Watchdog timer expired. Restarting process...")
        self.restart_process()

    def restart_process(self):
        try:
            # Get command line arguments of the current process
            args = sys.argv[:]
            args.insert(0, sys.executable)
            
            # Log restart attempt
            bt.logging.info(f"Attempting to restart with command: {' '.join(args)}")
            
            # Start new process
            subprocess.Popen(args)
            
            # Exit current process
            os._exit(0)
        except Exception as e:
            bt.logging.error(f"Failed to restart process: {str(e)}")
            # If restart fails, we'll exit anyway to avoid leaving the process in an unknown state
            os._exit(1)