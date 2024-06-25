# LOCAL
#python neurons/validator.py --netuid 1 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator-local --wallet.hotkey default --logging.debug --logging.trace

# MAIN
# python neurons/validator.py --netuid 30 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator-local --wallet.hotkey default --logging.debug --logging.trace

# TEST
python neurons/validator.py --netuid 181 --subtensor.network test --wallet.name validator --wallet.hotkey test2 --logging.debug --logging.trace
