#!/bin/bash

# Default values
LITE_NODE=false
NETWORK="test"

echo  "WARNING: If you are running in a containerized service (vast.ai, runpod, etc), or with a firewall, you may need to add the appropriate ports to the firewall rules.\n
        Ports: \n
        9944 - Websocket. This port is used by Bittensor. This port only accepts connections from localhost. Make sure this port is firewalled off from the public internet domain.\n
        9933 - RPC. This port should be opened but it is not used.\n
        30333 - p2p socket. This port should accept connections from other subtensor nodes on the internet. Make sure your firewall allows incoming traffic to this port.\n
        We assume that your default outgoing traffic policy is ACCEPT. If not, make sure that outbound traffic on port 30333 is allowed.\n
        If you decide to set up a firewall, be careful about blocking your ssh port (22) and losing access to your server.\n
        "




# Function to display usage information
usage() {
    echo "Usage: $0 [--lite-node] [--subtensor.network <test|main>]"
    echo "  --lite-node              Set up a lite node (optional)"
    echo "  --subtensor.network      Specify the network: test | main (default: test)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lite-node)
            LITE_NODE=true
            shift
            ;;
        --subtensor.network)
            if [[ $2 =~ ^(test|main|local)$ ]]; then
                NETWORK=$2
                shift 2
            else
                echo "Error: Invalid network specified."
                usage
            fi
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

echo "Setting up environment for network: $NETWORK"
if [ "$LITE_NODE" = true ]; then
    echo "Setting up as a lite node"
fi

# Update and install dependencies
sudo apt-get update 
sudo apt install -y build-essential
sudo apt-get install -y clang
sudo apt-get install -y curl 
sudo apt-get install -y git 
sudo apt-get install -y make
sudo apt install -y --assume-yes git clang curl libssl-dev protobuf-compiler
sudo apt install -y --assume-yes git clang curl libssl-dev llvm libudev-dev make protobuf-compiler

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Set up Rust
rustup default stable
rustup update
rustup target add wasm32-unknown-unknown
rustup toolchain install nightly
rustup target add --toolchain nightly wasm32-unknown-unknown

# Clone subtensor repository
cd $HOME
git clone https://github.com/opentensor/subtensor.git
cd subtensor
git checkout main

# Build subtensor
cargo build --release --features=runtime-benchmarks

# Set up the node based on the network
if [ "$LITE_NODE" = true ] && [ "$NETWORK" != "local" ]; then
    if [ "$NETWORK" = "main" ]; then
        echo "Setting up lite node for main network"
        ./target/release/node-subtensor --chain raw_spec.json --base-path /tmp/blockchain --sync=warp --execution wasm --wasm-execution compiled --port 30333 --max-runtime-instances 64 --rpc-max-response-size 2048 --rpc-cors all --rpc-port 9933 --bootnodes /ip4/13.58.175.193/tcp/30333/p2p/12D3KooWDe7g2JbNETiKypcKT1KsCEZJbTzEHCn8hpd4PHZ6pdz5 --no-mdns --in-peers 8000 --out-peers 8000 --prometheus-external --rpc-external
    elif [ "$NETWORK" = "test" ]; then
        echo "Setting up lite node for test network"
        ./target/release/node-subtensor --chain raw_testspec.json --base-path /tmp/blockchain --sync=warp --execution wasm --wasm-execution compiled --port 30333 --max-runtime-instances 64 --rpc-max-response-size 2048 --rpc-cors all --rpc-port 9933 --bootnodes /dns/bootnode.test.finney.opentensor.ai/tcp/30333/p2p/12D3KooWPM4mLcKJGtyVtkggqdG84zWrd7Rij6PGQDoijh1X86Vr --no-mdns --in-peers 8000 --out-peers 8000 --prometheus-external --rpc-external
    fi
else
    echo "Lite node setup is only available for main and test networks."
    echo "For local network or full node setup, please refer to the documentation for next steps."
fi

echo "Setup complete!"
# Need to install python 3.10, venv, etc. 
# npm, jq , pm2. 
# create venv named .venv in root/top level directory and activate
# cd back to bettensor and install dependencies (requirements.txt and -e .)
# verify everything.
# test that it works on 20.04 and 22.04 ubuntu (can use same contabo instance and reinstall the OS/reboot, or make one more and use that if you want to keep your testnet stuff) 
# new branch and pull request to auto_update