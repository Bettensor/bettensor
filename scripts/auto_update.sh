#!/bin/bash

SCRIPT_NAME=$1
DISABLE_AUTO_UPDATE=$2

if [ "$DISABLE_AUTO_UPDATE" = "true" ]; then
    echo "Auto-update is disabled. Running $SCRIPT_NAME without updates."
    exec $SCRIPT_NAME
    exit 0
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Auto-update enabled on branch: $current_branch"

while true; do
    git fetch
    local_hash=$(git rev-parse HEAD)
    remote_hash=$(git rev-parse origin/$current_branch)

    if [[ $local_hash != $remote_hash ]]; then
        echo "New updates detected. Pulling changes..."
        git pull origin $current_branch
        echo "Reinstalling dependencies..."
        pip install -e .
        echo "Restarting process..."
        exec $SCRIPT_NAME
        exit 0
    else
        echo "No updates found. Checking again in 10 minutes..."
    fi
    sleep 600
done