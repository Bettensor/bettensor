#!/bin/bash

# Change to the repository root directory
cd "$(dirname "$0")/.." || exit 1

echo "Starting auto_update.sh"
echo "Current working directory: $(pwd)"

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Auto-update enabled on branch: $current_branch"

chmod +x scripts/restart_pm2_processes.sh

cd "$PROJECT_ROOT" || exit 1
echo "Changed working directory to: $(pwd)"

# Check if this is the first run after update
FIRST_RUN_FLAG="/tmp/bettensor_first_run_after_update"

if [ ! -f "$FIRST_RUN_FLAG" ]; then
    echo "First run after update. Executing migration and setup processes..."
    
    if [ -f ./scripts/migrate.sh ]; then
        chmod +x ./scripts/migrate.sh
        echo "Running migrate.sh..."
        if ! bash ./scripts/migrate.sh; then
            echo "migrate.sh failed. Exiting."
            exit 1
        fi
    else
        echo "migrate.sh not found. Skipping setup process."
    fi

    # Create the flag file to indicate that the first run process has been completed
    touch "$FIRST_RUN_FLAG"
    
    echo "First run processes completed. Restarting the script..."
    exec "$0" "$@"
fi

update_and_restart() {
    echo "New updates detected. Stashing local changes..."
    git stash
    echo "Pulling changes..."
    if git pull origin $current_branch; then
        
        rm -f "$FIRST_RUN_FLAG"
        echo "Making scripts executable..."
        find ./scripts -type f -name "*.sh" -exec chmod +x {} \;
        echo "Making post-merge hook executable..."
        if [ ! -f .git/hooks/post-merge ]; then
            echo "#!/bin/bash" > .git/hooks/post-merge
            echo "./scripts/migrate.sh" >> .git/hooks/post-merge
            echo "pip install -e ." >> .git/hooks/post-merge
            echo "./scripts/restart_pm2_processes.sh" >> .git/hooks/post-merge
        fi
        chmod +x .git/hooks/post-merge
        echo "Triggering migration and setup process..."
        if [ -f ./scripts/migrate.sh ]; then
            chmod +x ./scripts/migrate.sh
            bash ./scripts/migrate.sh
        else
            echo "migrate.sh not found. Skipping setup process."
        fi  
        echo "Reinstalling dependencies..."
        if pip install -e .; then
            echo "Scheduling PM2 restart..."
            nohup bash -c "sleep 10 && $(pwd)/scripts/restart_pm2_processes.sh" > /tmp/pm2_restart.log 2>&1 &
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