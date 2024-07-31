import os
import time
import fcntl
import bittensor as bt

class DatabaseLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_fd = None

    def acquire(self, timeout=30):
        start_time = time.time()
        while True:
            try:
                self.lock_fd = open(self.lock_file, 'w')
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError:
                if time.time() - start_time > timeout:
                    bt.logging.warning(f"Timeout waiting for database lock")
                    return False
                time.sleep(1)

    def release(self):
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
