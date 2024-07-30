#!/bin/bash

# Source the user's profile to ensure all necessary environment variables are set
source ~/.profile

# Use the full path to PM2
PM2_PATH=$(which pm2)

echo "Restarting all PM2 processes..."
$PM2_PATH jlist | jq -r '.[] | .pm_id' | while read -r process_id; do
    echo "Attempting to restart process ID: $process_id"
    if $PM2_PATH restart "$process_id" --update-env; then
        echo "Successfully restarted process ID $process_id"
    else
        echo "Failed to restart process ID $process_id"
    fi
done

# Force PM2 to save the new process list
$PM2_PATH save

echo "PM2 restart process completed"

# Ensure this script exits
exit 0