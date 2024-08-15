#!/bin/bash

# migration.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


set -e

echo "Starting Bettensor migration process..."


export DB_NAME=bettensor
export DB_USER=root
export DB_PASSWORD=bettensor_password
export DB_HOST=localhost
export DB_PORT=5432


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
    if ! command -v redis-server &> /dev/null; then
        echo "Redis is not installed. Installing Redis..."
        sudo apt-get update
        sudo apt-get install -y redis-server
    fi

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

    if [ -f /etc/redis/redis.conf ]; then
        if ! grep -q "bind 0.0.0.0" /etc/redis/redis.conf; then
            echo "Modifying Redis configuration to allow connections from anywhere"
            sudo sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf
            sudo systemctl restart redis-server
        else
            echo "Redis configuration already allows connections from anywhere"
        fi
    else
        echo "Warning: Redis configuration file not found. Continuing with migration..."
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
    if ! command -v psql &> /dev/null
    then
        echo "PostgreSQL is not installed. Installing PostgreSQL..."
        sudo apt-get update
        sudo apt-get install -y postgresql postgresql-contrib
    else
        echo "PostgreSQL is already installed."
    fi

    # Start PostgreSQL service
    sudo systemctl start postgresql
    sudo systemctl enable postgresql

    # Configure PostgreSQL
    sudo -u postgres psql -c "CREATE DATABASE bettensor;" || echo "Database 'bettensor' already exists."
    sudo -u postgres psql -c "CREATE USER root WITH SUPERUSER PASSWORD 'bettensor_password';" || echo "User 'root' already exists."
    sudo -u postgres psql -c "ALTER USER root WITH SUPERUSER;"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bettensor TO root;"

    # Modify PostgreSQL configuration to allow root access
    sudo sed -i "s/local   all             postgres                                peer/local   all             postgres                                trust/" /etc/postgresql/*/main/pg_hba.conf
    sudo sed -i "s/local   all             all                                     peer/local   all             all                                     trust/" /etc/postgresql/*/main/pg_hba.conf

    # Allow connections from anywhere (only for development, not recommended for production)
    sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/*/main/postgresql.conf
    echo "host all all 0.0.0.0/0 md5" | sudo tee -a /etc/postgresql/*/main/pg_hba.conf

    # Restart PostgreSQL to apply changes
    sudo systemctl restart postgresql

    echo "PostgreSQL setup completed successfully"
}

# Function to update Python dependencies
update_python_deps() {
    echo "Updating Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install --no-cache-dir psycopg2-binary
    pip install --no-cache-dir torch==1.13.1
    pip install --no-cache-dir bittensor==6.9.3
}

# Function to perform backup and data migration
backup_and_migrate() {
    echo "Performing backup and data migration..."
    if ! "${SCRIPT_DIR}/backup_and_migrate.sh"; then
        echo "Warning: Backup and migration encountered an error. Please check the logs and try again."
        return 1
    fi
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
