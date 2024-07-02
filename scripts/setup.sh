# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced. Please run: source ./$(basename ${BASH_SOURCE[0]}) <flags>"
    exit 1
fi

# Check Ubuntu version
ubuntu_version=$(lsb_release -rs)

if [[ "$ubuntu_version" == "20.04" ]]; then
    echo "Running on Ubuntu 20.04"
    # Add any 20.04-specific adjustments here
elif [[ "$ubuntu_version" == "22.04" ]]; then
    echo "Running on Ubuntu 22.04"
else
    echo "This script is only tested on Ubuntu 20.04 and 22.04"
    echo "Your version: $ubuntu_version"
    echo "The script may not work correctly. Do you want to continue? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Exiting script"
        exit 1
    fi
fi

# Default values
LITE_NODE=false
NETWORK="test"
echo "Running setup..."
echo -e "WARNING: If you are running in a containerized service (vast.ai, runpod, etc), or with a firewall, 
         you may need to add the appropriate ports to the firewall rules:
        Ports:
        9944 - Websocket. This port is used by Bittensor. This port only accepts connections from localhost.
               Make sure this port is firewalled off from the public internet domain.
        9933 - RPC. This port should be opened but it is not used.
        30333 - p2p socket. This port should accept connections from other subtensor nodes on the internet. 
                Make sure your firewall allows incoming traffic to this port.
        We assume that your default outgoing traffic policy is ACCEPT. If not, make sure that outbound traffic on port 30333 is allowed.
        If you decide to set up a firewall, be careful about blocking your ssh port (22) and losing access to your server.
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
sudo apt install -y build-essential net-tools
sudo apt-get install -y clang curl git make 
sudo apt install -y --assume-yes git clang curl libssl-dev protobuf-compiler
sudo apt install -y --assume-yes git clang curl libssl-dev llvm libudev-dev make protobuf-compiler
sudo apt-get install -y python-is-python3

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

# Install Python 3.10
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Install npm, jq, and pm2
sudo apt-get install -y npm jq
sudo npm install -g pm2

# Clone Bettensor repository
cd $HOME
git clone https://github.com/bettensor/bettensor.git


# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

cd bettensor
# Install Bettensor dependencies
pip install -e .
pip install -r requirements.txt

# Verify installation
python -c "import bittensor; print(bittensor.__version__)"

echo "Bettensor setup complete!"
echo "Please ensure you are running Bittensor v6.9.3 as required for the current Beta version."
echo "You can now start mining or validating on the Bettensor subnet, after registering your miner/validator with the subnet."