#!/bin/bash

echo "Restarting all PM2 processes..."
pm2 jlist | jq -r '.[] | .pm_id' | while read -r process_id; do
    echo "Attempting to restart process ID: $process_id"
    if pm2 restart "$process_id" --update-env; then
        echo "Successfully restarted process ID $process_id"
    else
        echo "Failed to restart process ID $process_id"
    fi
done