#!/bin/bash

# Make auto_update.sh executable
chmod +x scripts/auto_update.sh

# Default neuron arguments
NEURON_ARGS="--netuid 1 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator-local --wallet.hotkey default --logging.debug --logging.trace"

# Parse arguments
DISABLE_AUTO_UPDATE=false

for arg in "$@"
do
    case $arg in
        --disable-auto-update)
        DISABLE_AUTO_UPDATE=true
        shift
        ;;
        *)
        # If it's not our custom flag, add it to NEURON_ARGS
        NEURON_ARGS="$NEURON_ARGS $arg"
        shift
        ;;
    esac
done

# If auto-update is not disabled, run the auto_update.sh script in the background
if [ "$DISABLE_AUTO_UPDATE" = false ]; then
    scripts/auto_update.sh "$0" false &
    AUTO_UPDATE_PID=$!
fi

# The actual validator command
python neurons/validator.py $NEURON_ARGS

# If auto-update is running, kill it when the main process exits
if [ "$DISABLE_AUTO_UPDATE" = false ]; then
    kill $AUTO_UPDATE_PID
fi
