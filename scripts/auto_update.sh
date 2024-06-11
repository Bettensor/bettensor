#!/bin/bash

# Auto-update script
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Auto-update checking enabled on branch: $current_branch"

while true; do
    git fetch
    local_hash=$(git rev-parse HEAD)
    remote_hash=$(git rev-parse origin/$current_branch)

    if [[ $local_hash != $remote_hash ]]; then
        echo "New updates detected. Pulling changes..."
        git pull origin $current_branch
        git checkout $current_branch
        echo "Reinstalling dependencies..."
        pip install -e .[validator]

        echo "Restarting PM2 process..."
        pm2 restart $INSTANCE
        echo "PM2 process $INSTANCE has been restarted."
    else
        echo "No updates found. Checking again in 10 minutes..."
    fi
    sleep 600  # Check every 10 minutes
done