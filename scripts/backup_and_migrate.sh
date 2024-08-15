#!/bin/bash

# backup_and_migrate.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source the .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Set default values if not provided in .env
DB_PATH=${DB_PATH:-"$PROJECT_ROOT/data/miner.db"}
BACKUP_DIR=${BACKUP_DIR:-"$PROJECT_ROOT/backups"}

echo "DB_PATH: $DB_PATH"
echo "BACKUP_DIR: $BACKUP_DIR"
echo "Current working directory: $(pwd)"

# Perform backup
if ! python -c "from bettensor.miner.utils.database_backup import trigger_backup; trigger_backup('$DB_PATH', '$BACKUP_DIR')"; then
    echo "Backup failed. Exiting."
    exit 1
fi

# Run migration script
if ! python "$PROJECT_ROOT/bettensor/miner/utils/migrate_to_postgres.py"; then
    echo "Migration failed. Exiting."
    exit 1
fi

echo "Backup and migration completed successfully."