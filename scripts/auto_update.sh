#!/bin/bash

# Change to the repository root directory
cd "$(dirname "$0")/.." || exit 1

echo "Starting auto_update.sh"
echo "Current working directory: $(pwd)"

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Auto-update enabled on branch: $current_branch"

chmod +x scripts/restart_pm2_processes.sh

update_and_restart() {
    echo "New updates detected. Stashing local changes..."
    git stash
    echo "Pulling changes..."
    if git pull origin $current_branch; then
        echo "Reinstalling dependencies..."
        if pip install -e .; then
            echo "Scheduling PM2 restart..."
            # Run the restart script detached from this process
            nohup bash -c "sleep 10 && $(pwd)/scripts/restart_pm2_processes.sh" > /tmp/pm2_restart.log 2>&1 &
            echo "PM2 restart scheduled. The script will exit now and restart shortly."
            # Force exit to ensure the process terminates
            exit 0
        else
            echo "Failed to install dependencies. Skipping restart."
            git stash pop
            return 1
        fi
    else
        echo "Failed to pull changes. Skipping update and restart."
        git stash pop
        return 1
    fi
}

while true; do
    echo "Fetching updates..."
    git fetch
    local_hash=$(git rev-parse HEAD)
    remote_hash=$(git rev-parse origin/$current_branch)

    echo "Local hash: $local_hash"
    echo "Remote hash: $remote_hash"

    if [[ $local_hash != $remote_hash ]]; then
        if update_and_restart; then
            echo "Update successful."
            sleep 120
        else
            echo "Update failed. Retrying in 5 minutes."
            sleep 300
        fi
    else
        echo "No updates found. Checking again in 2 minutes..."
        sleep 120
    fi
done