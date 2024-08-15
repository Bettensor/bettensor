#!/bin/bash

# migration.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


set -e

echo "Starting Bettensor migration process..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and install dependencies
install_dependencies() {
    echo "Checking and installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential net-tools clang curl git make libssl-dev protobuf-compiler llvm libudev-dev python-is-python3
}

# Function to install and configure Redis
setup_redis() {
    echo "Setting up Redis..."
    if systemctl is-active --quiet redis-server; then
        echo "Redis is already running"
    else
        if sudo systemctl start redis-server; then
            echo "Redis started successfully"
        else
            echo "Warning: Failed to start Redis server. Continuing with migration..."
        fi
    fi

    sudo systemctl enable redis-server || echo "Warning: Failed to enable Redis server. Continuing with migration..."

    if ! grep -q "bind 0.0.0.0" /etc/redis/redis.conf; then
        echo "Modifying Redis configuration to allow connections from anywhere"
        if sudo sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf; then
            if ! sudo systemctl restart redis-server; then
                echo "Warning: Failed to restart Redis server after configuration change. Continuing with migration..."
            fi
        else
            echo "Warning: Failed to modify Redis configuration. Continuing with migration..."
        fi
    else
        echo "Redis configuration already allows connections from anywhere"
    fi

    if systemctl is-active --quiet redis-server; then
        echo "Redis setup completed successfully"
    else
        echo "Warning: Redis setup encountered issues. Continuing with migration..."
    fi
}
# Function to backup the current database
backup_database() {
    echo "Backing up the current database..."
    pg_dump bettensor > bettensor_backup_$(date +%Y%m%d_%H%M%S).sql
}

# Function to install and configure PostgreSQL
setup_postgres() {
    echo "Setting up PostgreSQL..."
    if ! command_exists psql; then
        sudo apt-get install -y postgresql postgresql-contrib
    fi
    sudo systemctl enable postgresql
    sudo systemctl start postgresql

    # Set up PostgreSQL
    sudo -u postgres psql -c "CREATE DATABASE bettensor;"
    sudo -u postgres psql -c "CREATE USER root WITH SUPERUSER PASSWORD 'bettensor_password';"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bettensor TO root;"

    # Modify PostgreSQL configuration to allow root access
    sudo sed -i "s/local   all             postgres                                peer/local   all             postgres                                trust/" /etc/postgresql/*/main/pg_hba.conf
    sudo sed -i "s/local   all             all                                     peer/local   all             all                                     trust/" /etc/postgresql/*/main/pg_hba.conf

    sudo systemctl restart postgresql
}

# Function to update Python dependencies
update_python_deps() {
    echo "Updating Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install --no-cache-dir psycopg2-binary
    pip install --no-cache-dir torch
}

# Function to perform backup and data migration
backup_and_migrate() {
    echo "Performing backup and data migration..."
    "${SCRIPT_DIR}/backup_and_migrate.sh"
}


# Function to update configuration files
update_config() {
    echo "Updating configuration files..."
    # Update any necessary configuration files to use PostgreSQL and Redis
    # This might involve modifying config files or environment variables
}

# Main migration process
main() {
    backup_and_migrate || echo "Warning: Backup and migration step failed. Continuing..."
    install_dependencies || echo "Warning: Dependencies installation failed. Continuing..."
    setup_redis || echo "Warning: Redis setup failed. Continuing..."
    setup_postgres || echo "Warning: PostgreSQL setup failed. Continuing..."
    update_python_deps || echo "Warning: Python dependencies update failed. Continuing..."
    update_config || echo "Warning: Configuration update failed. Continuing..."

    echo "Migration completed. Some steps might have encountered issues."
    echo "Please review the logs and manually address any reported warnings or errors."
    echo "You may need to restart your Bettensor miner to apply the changes."
}

# Run the main function
main
