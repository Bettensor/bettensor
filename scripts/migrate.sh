#!/bin/bash

# migration.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

set -e

echo "Starting Bettensor migration process..."

# Source the .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Determine if this is a validator, miner, or both
IS_VALIDATOR=false
IS_MINER=false

if [ -f "$PROJECT_ROOT/data/validator.db" ]; then
    IS_VALIDATOR=true
    echo "Detected validator configuration."
fi

if [ -f "$PROJECT_ROOT/data/miner.db" ]; then
    IS_MINER=true
    echo "Detected miner configuration."
fi

# If it's only a validator, skip the migration
if [ "$IS_VALIDATOR" = true ] && [ "$IS_MINER" = false ]; then
    echo "Only validator configuration detected. Skipping database migration."
    exit 0
fi


export DB_NAME=${DB_NAME:-bettensor}
export DB_USER=${DB_USER:-root}
export DB_PASSWORD=${DB_PASSWORD:-bettensor_password}
export DB_HOST=${DB_HOST:-localhost}
export DB_PORT=${DB_PORT:-5432}
export SQLITE_DB_PATH=${SQLITE_DB_PATH:-"$PROJECT_ROOT/data/miner.db"}
export BACKUP_DIR=${BACKUP_DIR:-"$PROJECT_ROOT/backups"}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and install dependencies
install_dependencies() {
    echo "Checking and installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential net-tools clang curl git make libssl-dev protobuf-compiler llvm libudev-dev python-is-python3 postgresql postgresql-contrib
}

# Install dependencies
install_dependencies

# Run the Python migration script
if [ "$IS_MINER" = true ]; then
    echo "Running migration for miner database..."
    python3 "$PROJECT_ROOT/bettensor/miner/utils/migrate_database.py"
else
    echo "No miner database detected. Skipping miner-specific migration."
fi

echo "Migration process completed. Please check the logs for any errors."