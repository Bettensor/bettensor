# LOCAL 
python neurons/miner.py --netuid 1 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner-local --wallet.hotkey default --logging.debug --logging.trace --axon.port 12345 --validator_min_stake 10

# MAIN 
#python neurons/miner.py --netuid 30 --subtensor.network finney --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner-local --wallet.hotkey default --logging.debug --logging.trace --axon.port 12345 --validator_min_stake 1000

# TEST 
#python neurons/miner.py --netuid 181 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug --logging.trace --axon.port 12345 --validator_min_stake 10
