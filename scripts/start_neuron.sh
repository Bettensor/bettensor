#!/bin/bash

# Check and set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"

if [ "$PWD" != "$REPO_ROOT" ]; then
    echo "Changing working directory to $REPO_ROOT"
    cd "$REPO_ROOT" || { echo "Failed to change directory. Exiting."; exit 1; }
fi

# Make sure all necessary scripts are executable
chmod +x scripts/*.sh
chmod +x neurons/*.py

# Default neuron arguments
DEFAULT_NEURON_ARGS=""
DISABLE_AUTO_UPDATE="false"
NEURON_TYPE=""
NETWORK=""
WALLET_NAME=""
WALLET_HOTKEY=""
AXON_PORT=""
VALIDATOR_MIN_STAKE=""
LOGGING_LEVEL=""

# Function to prompt for user input
prompt_for_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    if [ -z "${!var_name}" ]; then
        read -p "$prompt [$default]: " user_input
        eval $var_name="${user_input:-$default}"
    fi
}

# Function to prompt for yes/no input
prompt_yes_no() {
    local prompt="$1"
    local var_name="$2"
    while true; do
        read -p "$prompt [y/n]: " yn
        case $yn in
            [Yy]* ) eval $var_name="true"; break;;
            [Nn]* ) eval $var_name="false"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to start Flask server
start_flask_server() {
    echo "Starting Flask server..."
    pm2 start --name "flask-server" python -- \
        -m bettensor.miner.interfaces.miner_interface_server \
        --host "127.0.0.1" --port 5000
    
    sleep 2  # Give the server a moment to start

    # Check if the server started successfully
    if pm2 list | grep -q "flask-server"; then
        echo "Flask server started successfully."
        pm2 logs "flask-server" --lines 20 --nostream
    else
        echo "Failed to start Flask server. Check logs for details."
        pm2 logs "flask-server" --lines 20 --nostream
    fi
}

# Function to start auto-updater
start_auto_updater() {
    echo "Starting auto-updater..."
    pm2 start --name "auto-updater" \
        --cwd "$REPO_ROOT" \
        bash \
        -- scripts/auto_update.sh
    
    sleep 5  # Give the process a moment to start

    if pm2 list | grep -q "auto-updater"; then
        echo "Auto-updater started successfully."
        pm2 logs "auto-updater" --lines 20 --nostream
    else
        echo "Failed to start auto-updater. Check logs for details."
        pm2 logs "auto-updater" --lines 20 --nostream
    fi
}

# Prompt for neuron type if not specified
prompt_for_input "Enter neuron type (miner/validator)" "miner" "NEURON_TYPE"

# Prompt for network if not specified
prompt_for_input "Enter network (local/test/finney)" "finney" "NETWORK"
case $NETWORK in
    test)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.network test --netuid 181"
        ;;
    finney)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.network finney --netuid 30"
        ;;
    local)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.network local --netuid 1"
        ;;
    *)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.network $NETWORK"
        ;;
esac

# Prompt for wallet name and hotkey if not specified
prompt_for_input "Enter wallet name" "default" "WALLET_NAME"
prompt_for_input "Enter wallet hotkey" "default" "WALLET_HOTKEY"
DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --wallet.name $WALLET_NAME --wallet.hotkey $WALLET_HOTKEY"

# Miner-specific arguments
if [ "$NEURON_TYPE" = "miner" ]; then
    prompt_for_input "Enter validator_min_stake" "1000" "VALIDATOR_MIN_STAKE"
    DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --validator_min_stake $VALIDATOR_MIN_STAKE"
    prompt_for_input "Enter axon port" "12345" "AXON_PORT"
    DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --axon.port $AXON_PORT"
fi

# Prompt for logging level if not specified
prompt_for_input "Enter logging level (info/debug/trace)" "debug" "LOGGING_LEVEL"
DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --logging.$LOGGING_LEVEL"

# Prompt for disabling auto-update
prompt_yes_no "Do you want to disable auto-update? Warning: this will apply to all running neurons" "DISABLE_AUTO_UPDATE"

# Check for existing miners and prompt to start another
MINER_COUNT=$(pm2 list | grep -c "miner")
if [ $MINER_COUNT -gt 0 ]; then
    prompt_yes_no "There are already $MINER_COUNT miner(s) running. Do you want to start another?" "START_ANOTHER_MINER"
    if [ "$START_ANOTHER_MINER" = "false" ]; then
        echo "Exiting without starting a new miner."
        exit 0
    fi
fi

# Start the neuron with PM2
MINER_NAME="miner$MINER_COUNT"
echo "Starting $NEURON_TYPE with arguments: $DEFAULT_NEURON_ARGS"
pm2 start --name "$MINER_NAME" python -- ./neurons/$NEURON_TYPE.py $DEFAULT_NEURON_ARGS

# Check if the neuron started successfully
if pm2 list | grep -q "$MINER_NAME"; then
    echo "$NEURON_TYPE started successfully with instance name: $MINER_NAME"
else
    echo "Failed to start $NEURON_TYPE. Check logs for details."
    pm2 logs "$MINER_NAME" --lines 20
    exit 1
fi

# Start additional services only for miners
if [ "$NEURON_TYPE" = "miner" ]; then
    # Start Flask server
    start_flask_server
fi

# Start auto-updater if not disabled
if [ "$DISABLE_AUTO_UPDATE" = "false" ]; then
    start_auto_updater
else
    echo "Auto-updater is disabled."
fi

# Save the PM2 process list
pm2 save --force

# Display running processes
echo "Current PM2 processes:"
pm2 list

# Display logs for all processes
for process in "$MINER_NAME" "flask-server" "auto-updater"; do
    if pm2 list | grep -q "$process"; then
        echo "Logs for $process:"
        pm2 logs "$process" --lines 20 --nostream
        echo ""
    fi
done