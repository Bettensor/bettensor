#!/bin/bash

echo "Starting auto_update.sh"

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Auto-update enabled on branch: $current_branch"

update_and_restart() {
    echo "New updates detected. Pulling changes..."
    if git pull origin $current_branch; then
        echo "Reinstalling dependencies..."
        if pip install -e .; then
            echo "Restarting all PM2 processes..."
            pm2 list --no-color | awk 'NR>2 {print $4}' | while read -r process_id; do
                echo "Read process ID: '$process_id'"
                if [ ! -z "$process_id" ]; then
                    echo "Attempting to restart process ID: $process_id"
                    if pm2 restart "$process_id" --update-env; then
                        echo "Successfully restarted process ID $process_id"
                    else
                        echo "Failed to restart process ID $process_id"
                    fi
                else
                    echo "Skipping empty process ID"
                fi
            done
            return 0
        else
            echo "Failed to install dependencies. Skipping restart."
            return 1
        fi
    else
        echo "Failed to pull changes. Skipping update and restart."
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