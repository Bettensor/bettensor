import os
import shutil
import subprocess
from datetime import datetime
import bittensor as bt

def backup_database(db_path, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if db_path.endswith('.db'):
        # SQLite backup
        backup_file = os.path.join(backup_dir, f"bettensor_backup_{timestamp}.db")
        shutil.copy2(db_path, backup_file)
        bt.logging.info(f"SQLite database backed up to {backup_file}")
    else:
        # PostgreSQL backup
        backup_file = os.path.join(backup_dir, f"bettensor_backup_{timestamp}.sql")
        try:
            subprocess.run(
                f"pg_dump -h localhost -U bettensor bettensor > {backup_file}",
                shell=True,
                check=True
            )
            bt.logging.info(f"PostgreSQL database backed up to {backup_file}")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Failed to backup PostgreSQL database: {e}")
            return None

    return backup_file

def trigger_backup(db_path, backup_dir):
    bt.logging.info("Triggering database backup before update")
    backup_file = backup_database(db_path, backup_dir)
    if backup_file:
        bt.logging.info(f"Database backed up successfully to {backup_file}")
    else:
        bt.logging.error("Failed to backup database")
