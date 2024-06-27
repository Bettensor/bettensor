# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from argparse import ArgumentParser
import sys
import os
import torch
import traceback
import time
import bittensor as bt
from bettensor import __version__ as version
from bettensor.miner.bettensor_miner import BettensorMiner


# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Get the grandparent directory
grandparent_dir = os.path.dirname(parent_dir)

# Get the great grandparent directory
great_grandparent_dir = os.path.dirname(grandparent_dir)

# Add parent, grandparent, and great grandparent directories to sys.path
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(great_grandparent_dir)

# Optional: Print sys.path to verify the directories have been added
print(sys.path)


def main(miner: BettensorMiner):
    """
    This function executes the main miner loop. The miner is configured
    upon the initialization of the miner. If you want to change the
    miner configuration, please adjust the initialization parameters.
    """

    # Link the miner to the Axon
    axon = bt.axon(wallet=miner.wallet, config=miner.neuron_config)
    bt.logging.info(f"Linked miner to Axon: {axon}")

    # Attach the miner functions to the Axon
    axon.attach(
        forward_fn=miner.forward,
        blacklist_fn=miner.blacklist,
        priority_fn=miner.priority,
    )
    bt.logging.info(f"Attached functions to Axon: {axon}")

    # Pass the Axon information to the network
    axon.serve(netuid=miner.neuron_config.netuid, subtensor=miner.subtensor)

    bt.logging.info(
        f"Axon {miner.forward} served on network: {miner.neuron_config.subtensor.chain_endpoint} with netuid: {miner.neuron_config.netuid}"
    )
    # Activate the Miner on the network
    axon.start()
    bt.logging.info(f"Axon started on port: {miner.neuron_config.axon.port}")
    bt.logging.info(f"Axon: {axon}")
    # Step 7: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(
        "Miner has been initialized and we are connected to the network. Start main loop."
    )

    # When we init, set last_updated_block to current_block
    miner.last_updated_block = miner.subtensor.block
    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            if miner.step % 20 == 0:
                # if miner.step % 300 == 0:
                # Check if the miners hotkey is on the remote blacklist
                # miner.check_remote_blacklist()

                if miner.step % 600 == 0:
                    bt.logging.debug(
                        f"Syncing metagraph: {miner.metagraph} with subtensor: {miner.subtensor}"
                    )

                    miner.metagraph.sync(subtensor=miner.subtensor)

                miner.metagraph = miner.subtensor.metagraph(miner.neuron_config.netuid)
                log = (
                    f"Version:{version} | "
                    f"Blacklist:{miner.hotkey_blacklisted} | "
                    f"Step:{miner.step} | "
                    f"Block:{miner.metagraph.block.item()} | "
                    f"Stake:{miner.metagraph.S[miner.miner_uid]} | "
                    f"Rank:{miner.metagraph.R[miner.miner_uid]} | "
                    f"Trust:{miner.metagraph.T[miner.miner_uid]} | "
                    f"Consensus:{miner.metagraph.C[miner.miner_uid] } | "
                    f"Incentive:{miner.metagraph.I[miner.miner_uid]} | "
                    f"Emission:{miner.metagraph.E[miner.miner_uid]}"
                )

                bt.logging.info(log)
                bt.logging.info(f"Miner UID: {miner.miner_uid}")

                bt.logging.warning(f"TESTING AUTO UPDATE!!")

            miner.step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--netuid", type=int, default=14, help="The chain subnet uid")
    parser.add_argument(
        "--logging.logging_dir",
        type=str,
        default="/var/log/bittensor",
        help="Provide the log directory",
    )

    parser.add_argument(
        "--miner_set_weights",
        type=str,
        default="False",
        help="Determines if miner should set weights or not",
    )

    parser.add_argument(
        "--validator_min_stake",
        type=float,
        default=10000.0,
        help="Determine the minimum stake the validator should have to accept requests",
    )

    # Create a miner based on the Class definitions
    subnet_miner = BettensorMiner(parser=parser)

    main(subnet_miner)
