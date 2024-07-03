#!/bin/bash

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

# Function to check if a port is in use
is_port_in_use() {
    netstat -tuln | grep -q ":$1 "
}

# Function to get an available port
get_available_port() {
    local port=$1
    while is_port_in_use $port; do
        echo "Port $port is already in use."
        port=$(prompt_for_input "Enter a different port" $((port + 1)) "AXON_PORT")
    done
    echo $port
}

# Function to check if a neuron is already running
check_existing_neurons() {
    local count=$(pm2 list | grep -c "$NEURON_TYPE")
    if [ $count -gt 0 ]; then
        read -p "There are already $count $NEURON_TYPE(s) running. Do you want to start another one? (y/n) " answer
        if [[ $answer != [Yy]* ]]; then
            echo "Operation cancelled."
            exit 0
        fi
    fi
}

# Function to get the next available instance number
get_next_instance_number() {
    local type=$1
    local max_num=-1
    local instances=$(pm2 jlist | jq -r '.[] | select(.name | startswith("'$type'")) | .name' | grep -oP '\d+$' | sort -n)
    
    if [ -z "$instances" ]; then
        echo 0
    else
        local last_num=$(echo "$instances" | tail -n 1)
        echo $((last_num + 1))
    fi
}

# Function to check if auto-updater is already running
is_auto_updater_running() {
    pm2 list | grep -q "auto-updater"
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

# Function to check if lite node is running
is_lite_node_running() {
    pgrep -f "substrate.*--chain bittensor" > /dev/null
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --miner)
            NEURON_TYPE="miner"
            shift
            ;;
        --validator)
            NEURON_TYPE="validator"
            shift
            ;;
        --network)
            NETWORK="$2"
            shift 2
            ;;
        --wallet.name)
            WALLET_NAME="$2"
            shift 2
            ;;
        --wallet.hotkey)
            WALLET_HOTKEY="$2"
            shift 2
            ;;
        --axon.port)
            AXON_PORT="$2"
            shift 2
            ;;
        --validator_min_stake)
            VALIDATOR_MIN_STAKE="$2"
            shift 2
            ;;
        --disable-auto-update)
            DISABLE_AUTO_UPDATE="true"
            shift
            ;;
        --logging.level)
            LOGGING_LEVEL="$2"
            shift 2
            ;;
        --subtensor.chain_endpoint)
            DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.chain_endpoint $2"
            shift 2
            ;;
        *)
            DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS $1"
            shift
            ;;
    esac
done

# Prompt for neuron type if not specified
prompt_for_input "Enter neuron type (miner/validator)" "miner" "NEURON_TYPE"

# Check for existing neurons
check_existing_neurons

# Prompt for network if not specified
prompt_for_input "Enter network (local/test/main)" "test" "NETWORK"
case $NETWORK in
    test)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.network test --netuid 181"
        ;;
    main)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.network finney --netuid 30"
        ;;
    *)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.network $NETWORK"
        ;;
esac

# Check if lite node is running and add chain_endpoint if it is
if is_lite_node_running; then
    DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --subtensor.chain_endpoint ws://127.0.0.1:9946"
fi

# Prompt for wallet name and hotkey if not specified
prompt_for_input "Enter wallet name" "default" "WALLET_NAME"
prompt_for_input "Enter wallet hotkey" "default" "WALLET_HOTKEY"
DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --wallet.name $WALLET_NAME --wallet.hotkey $WALLET_HOTKEY"

# Prompt for axon port and validator_min_stake if miner and not specified
if [ "$NEURON_TYPE" = "miner" ]; then
    prompt_for_input "Enter axon port" "12345" "AXON_PORT"
    AXON_PORT=$(get_available_port $AXON_PORT)
    prompt_for_input "Enter validator_min_stake" "0" "VALIDATOR_MIN_STAKE"
    DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --axon.port $AXON_PORT --validator_min_stake $VALIDATOR_MIN_STAKE"
fi

# Prompt for logging level if not specified
prompt_for_input "Enter logging level (info/debug/trace)" "debug" "LOGGING_LEVEL"
case $LOGGING_LEVEL in
    info|debug|trace)
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --logging.$LOGGING_LEVEL"
        ;;
    *)
        echo "Invalid logging level. Using default (debug)."
        DEFAULT_NEURON_ARGS="$DEFAULT_NEURON_ARGS --logging.debug"
        ;;
esac

# Prompt for disabling auto-update if not specified
if [ "$DISABLE_AUTO_UPDATE" = "false" ]; then
    prompt_yes_no "Do you want to disable auto-update? Warning: this will apply to all running neurons" "DISABLE_AUTO_UPDATE"
fi

# Generate a unique name for this instance
INSTANCE_NUMBER=$(get_next_instance_number $NEURON_TYPE)
INSTANCE_NAME="${NEURON_TYPE}${INSTANCE_NUMBER}"

# Handle auto-updater
if [ "$DISABLE_AUTO_UPDATE" = "false" ]; then
    if ! is_auto_updater_running; then
        pm2 start scripts/auto_update.sh --name "auto-updater"
        echo "Auto-updater started."
    else
        echo "Auto-updater is already running."
    fi
else
    if is_auto_updater_running; then
        pm2 stop auto-updater
        pm2 delete auto-updater
        echo "Auto-updater has been stopped and removed."
    else
        echo "Auto-updater is not running."
    fi
fi

# Start the neuron with PM2
echo "Starting $NEURON_TYPE with arguments: $DEFAULT_NEURON_ARGS"
pm2 start --name "$INSTANCE_NAME" python -- neurons/$NEURON_TYPE.py $DEFAULT_NEURON_ARGS

echo "$NEURON_TYPE started successfully with instance name: $INSTANCE_NAME"


