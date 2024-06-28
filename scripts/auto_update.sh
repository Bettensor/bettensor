#!/bin/bash

echo "Starting auto_update.sh"

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Auto-update enabled on branch: $current_branch"

while true; do
    echo "Fetching updates..."
    git fetch
    local_hash=$(git rev-parse HEAD)
    remote_hash=$(git rev-parse origin/$current_branch)

    echo "Local hash: $local_hash"
    echo "Remote hash: $remote_hash"

    if [[ $local_hash != $remote_hash ]]; then
        echo "New updates detected. Pulling changes..."
        git pull origin $current_branch
        echo "Reinstalling dependencies..."
        pip install -e .
        echo "Restarting all neuron processes..."
        pm2 restart all --update-env
    else
        echo "No updates found. Checking again in 2 minutes..."
    fi
    sleep 120
done