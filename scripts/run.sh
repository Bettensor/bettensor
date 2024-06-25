#!/bin/bash
declare -A args


check_runtime_environment() {
    if ! python --version &>/dev/null; then
        echo "ERROR: Python is not available. Make sure Python is installed and venv has been activated."
        exit 1
    fi

    # Get Python version and check if it meets the minimum requirement of 3.10
    if ! python -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "ERROR: Python version must be at least 3.10"
        exit 1
    fi

    echo "The installed python version meets the minimum requirement (3.10)."

    # Check that the required packages are installed. These should be bundled with the OS and/or Python version. 
    # If they do not exists, they should be installed manually. We do not want to install these in the run script,
    # as it could mess up the local system

    package_list=("libssl-dev" "python"${values[0]}"."${values[1]}"-dev")

    error=0
    for package_name in "${package_list[@]}"; do
        if ! dpkg -l | grep -q -w "^ii  $package_name"; then
            echo "ERROR: $package_name is not installed. Please install it manually."
            error=1
        fi
    done

    if [[ $error -eq 1 ]]; then
        exit 1
    fi

    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Virtual environment is activated: $VIRTUAL_ENV"
    else
        echo "WARNING: Virtual environment is not activated. It is recommended to run this script in a python venv."
    fi
}

parse_arguments() {

    while [[ $# -gt 0 ]]; do
        if [[ $1 == "--"* ]]; then
            arg_name=${1:2}  # Remove leading "--" from the argument name

            # Special handling for logging argument
            if [[ "$arg_name" == "logging"* ]]; then
                shift
                if [[ $1 != "--"* ]]; then
                    IFS='.' read -ra parts <<< "$arg_name"
                    args[${parts[0]}]=${parts[1]}
                fi
            else
                shift
                args[$arg_name]="$1"  # Assign the argument value to the argument name
            fi
        fi
        shift
    done

    for key in "${!args[@]}"; do
        echo "Argument: $key, Value: ${args[$key]}"
    done
}

pull_repo_and_checkout_branch() {
    local branch="${args['branch']}"

    # Pull the latest repository
    git pull --all

    # Change to the specified branch if provided
    if [[ -n "$branch" ]]; then
        echo "Switching to branch: $branch"
        git checkout "$branch" || { echo "Branch '$branch' does not exist."; exit 1; }
    fi

    local current_branch=$(git symbolic-ref --short HEAD)
    git fetch &>/dev/null  # Silence output from fetch command
    if ! git rev-parse --quiet --verify "origin/$current_branch" >/dev/null; then
        echo "You are using a branch that does not exists in remote. Make sure your local branch is up-to-date with the latest version in the main branch. Auto-updater will not be enabled."
    fi
}

install_packages() {
    local cfg_version=$(grep -oP 'version\s*=\s*\K[^ ]+' setup.cfg)
    local installed_version=$(pip show healthi | grep -oP 'Version:\s*\K[^ ]+')
    local profile="${args['profile']}"

    # Load dotenv configuration
    DOTENV_FILE=".env"
    if [ -f "$DOTENV_FILE" ]; then
        # Load environment variables from .env file
        export $(grep -v '^#' $DOTENV_FILE | xargs)
        echo "Environment variables loaded from $DOTENV_FILE"
    fi

    # if [[ "$cfg_version" == "$installed_version" ]]; then
    #     echo "Subnet versions "$cfg_version" and "$installed_version" are matching: No installation is required."
    # else

    if [ "$profile" == "miner" ]; then
        echo "Installing python package with pip with miner extras"
        pip install -e .[miner]
    elif [ "$profile" == "validator" ]; then
        echo "Installing python package with pip with validator extras"
        pip install -e .[validator]
    else
        echo "Unable to determine profile. Exiting."
        exit 1
    fi

    # fi

    # Uvloop re-implements asyncio module which breaks bittensor. It is
    # not needed by the default implementation of the
    # subnet, so we can uninstall it.
    if pip show uvloop &>/dev/null; then
        echo "Uninstalling conflicting module uvloop"
        pip uninstall -y uvloop
    fi

    echo "All python packages are installed"
}


generate_pm2_launch_file() {
    echo "Generating PM2 launch file"
    local cwd=$(pwd)
    local profile="${args['profile']}"
    local neuron_script="${cwd}/neurons/${profile}.py"
    local interpreter="${VIRTUAL_ENV}/bin/python"
    local branch="${args['branch']}"
    local update_interval="${args['update_interval']}"

    # Script arguments
    local netuid="${args['netuid']}"
    local subtensor_chain_endpoint="${args['subtensor.chain_endpoint']}"
    local wallet_name="${args['wallet.name']}"
    local wallet_hotkey="${args['wallet.hotkey']}"
    local logging_value="${args['logging']}"
    local subtensor_network="${args['subtensor.network']}"
    local axon_port="${args['axon.port']}"
    local axon_ip="${args['axon.ip']}"
    local wallet_path="${args['wallet.path']}"
    local name="${args['name']}"
    local max_memory_restart="${args['max_memory_restart']}"
    local miner_set_weights="${args['miner_set_weights']}"
    local validator_min_stake="${args['validator_min_stake']}"

    # Construct argument list for the neuron
    if [[ -z "$netuid" || -z "$wallet_name" || -z "$wallet_hotkey" || -z "$name" || -z  "$max_memory_restart" ]]; then
        echo "name, max_memory_restart, netuid, wallet.name, and wallet.hotkey are mandatory arguments."
        exit 1
    fi

    local args="--netuid $netuid --wallet.name $wallet_name --wallet.hotkey $wallet_hotkey"

    if [[ -n "$subtensor_chain_endpoint" ]]; then
        args+=" --subtensor.chain_endpoint $subtensor_chain_endpoint"
    fi

    if [[ -n "$subtensor_network" ]]; then
        args+=" --subtensor.network $subtensor_network"
    fi

    if [[ -n "$axon_port" ]]; then
        args+=" --axon.port $axon_port"
    fi

    if [[ -n "$axon_ip" ]]; then
        args+=" --axon.ip $axon_ip"
    fi

    if [[ -n "$wallet_path" ]]; then
        args+=" --wallet.path $wallet_path"
    fi

    if [[ -n "$miner_set_weights" ]]; then
        args+=" --miner_set_weights $miner_set_weights"
    fi

    if [[ -n "$logging_value" ]]; then
        args+=" --logging.$logging_value"
    fi

    if [[ -n "$validator_min_stake" ]]; then 
        args+=" --validator_min_stake $validator_min_stake"
    fi

    cat <<EOF > ${name}.config.js
module.exports = {
    apps: [
        {
            "name"                  : "${name}",
            "script"                : "${neuron_script}",
            "interpreter"           : "${interpreter}",
            "args"                  : "${args}",
            "max_memory_restart"    : "${max_memory_restart}"
        }
    ]
}
EOF
}

generate_pm2_auto_update_config() {
    local cwd=$(pwd)
    local name="${args['name']}"
    cat <<EOF > auto_update.config.js
module.exports = {
    apps : [{
        name   : "auto-updater",
        script : "${cwd}/scripts/auto_update.sh",
        interpreter : "/bin/bash",
        env: {
            INSTANCE: "${name}"
        }
    }]
}
EOF
}

launch_pm2_instance() {
    local name="${args['name']}"
    eval "pm2 start ${name}.config.js"
}

launch_pm2_auto_update() {
    pm2 start auto_update.config.js
}

echo "### START OF EXECUTION ###"
# Parse arguments and assign to associative array
parse_arguments "$@"

profile="${args['profile']}"
install_only="${args['install_only']}"

check_runtime_environment
echo "Python venv checks completed. Sleeping 2 seconds."
sleep 2
pull_repo_and_checkout_branch
echo "Repo pulled and branch checkout done. Sleeping 2 seconds."
sleep 2
install_packages
echo "Installation done. Sleeping 2 seconds."
sleep 2


if [[ "$install_only" == 1 ]]; then
    echo "Installation done. PM2 ecosystem files not created and PM2 instance was not launched as install_only is set to True."
else
    echo "Generating PM2 ecosystem file"
    generate_pm2_launch_file
    echo "Launching PM instance"
    launch_pm2_instance
fi

if [[ "$profile" == "validator" ]]; then
    generate_pm2_auto_update_config
    echo "Generating PM2 auto-update file"
    sleep 2
    launch_pm2_auto_update
    echo "Lanuching auto-updater."
fi