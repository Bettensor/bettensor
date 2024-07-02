#!/bin/bash

echo "Starting auto_update.sh"

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
            # Run the restart script in the background after a short delay
            (sleep 10 && bash "$(pwd)/scripts/restart_pm2_processes.sh") &
            echo "PM2 restart scheduled. The script will exit now and restart shortly."
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