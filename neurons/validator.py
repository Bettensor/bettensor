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

import secrets
import time
import sys
import os

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
from bettensor.protocol import GameData

# Bittensor
import bittensor as bt
import copy
from copy import deepcopy
# Bittensor Validator Template:
import bettensor
from bettensor.validator import forward
from uuid import UUID
from argparse import ArgumentParser
import sqlite3
# import base validator class which takes care of most of the boilerplate
from bettensor.base.validator import BaseValidatorNeuron
# need to import the right protocol(s) here
from bettensor.validator.bettensor_validator import BettensorValidator
from bettensor import protocol
#from update_games import update_games
from datetime import datetime

def main(validator: BettensorValidator):
    
    while True:
        try:
            # Periodically sync subtensor status and save the state file
            if validator.step % 5 == 0:
                # Sync metagraph
                try:
                    validator.metagraph = validator.sync_metagraph(
                        validator.metagraph, validator.subtensor
                    )
                    bt.logging.debug(f'Metagraph synced: {validator.metagraph}')
                except TimeoutError as e:
                    bt.logging.error(f"Metagraph sync timed out: {e}")

                # Update local knowledge of the hotkeys
                validator.check_hotkeys()

                # Save state
                validator.save_state()

            if validator.step % 20 == 0:
                pass
                # Update local knowledge of blacklisted miner hotkeys
                #validator.check_blacklisted_miner_hotkeys()

            # Get all axons
            all_axons = validator.metagraph.axons
            bt.logging.trace(f"All axons: {all_axons}")

            # If there are more axons than scores, append the scores list
            if len(validator.metagraph.uids.tolist()) > len(validator.scores):
                bt.logging.info(
                    f"Discovered new Axons, current scores: {validator.scores}"
                )
                validator.scores = torch.cat(
                    (
                        validator.scores,
                        torch.zeros(
                            (
                                len(validator.metagraph.uids.tolist())
                                - len(validator.scores)
                            ),
                            dtype=torch.float32,
                        ),
                    )
                )
                bt.logging.info(f"Updated scores, new scores: {validator.scores}")
            # Get list of UIDs to query
            (
                uids_to_query,
                list_of_uids,
                blacklisted_uids,
                uids_not_to_query,
            ) = validator.get_uids_to_query(all_axons=all_axons)
            if not uids_to_query:
                bt.logging.warning(f"UIDs to query is empty: {uids_to_query}")


            # Broadcast query to valid Axons
            nonce = secrets.token_hex(24)
            timestamp = str(int(time.time()))
            data_to_sign = f'{nonce}{timestamp}'
            current_timestamp = datetime.now().isoformat()
            db_path = '/root/bettensor/bettensor/utils/games.db' #os.path.join(os.path.dirname(__file__), 'bettensor', 'utils', 'games.db')
            print(f"Attempting to open database at: {db_path}")
            if not os.path.exists(db_path):
                            raise FileNotFoundError(f"Database file not found at path: {db_path}")

            responses = validator.dendrite.query(
                uids_to_query,
                GameData.create(current_timestamp=current_timestamp, db_path=db_path),
                timeout=validator.timeout,
                deserialize=True,
            )

            # Process blacklisted UIDs (set scores to 0)
            for uid in blacklisted_uids:
                bt.logging.debug(f'Setting score for blacklisted UID: {uid}. Old score: {validator.scores[uid]}')
                validator.scores[uid] = (
                    validator.neuron_config.alpha * validator.scores[uid]
                    + (1 - validator.neuron_config.alpha) * 0.0
                )
                bt.logging.debug(f'Set score for blacklisted UID: {uid}. New score: {validator.scores[uid]}')

            # Process UIDs we did not query (set scores to 0)
            for uid in uids_not_to_query:
                bt.logging.trace(
                    f"Setting score for not queried UID: {uid}. Old score: {validator.scores[uid]}"
                )
                validator.scores[uid] = (
                    validator.neuron_config.alpha * validator.scores[uid]
                    + (1 - validator.neuron_config.alpha) * 0.0
                )
                bt.logging.trace(
                    f"Set score for not queried UID: {uid}. New score: {validator.scores[uid]}"
                )

            # Log the results for monitoring purposes.
            if all(item.output is None for item in responses):
                bt.logging.info("Received empty response from all miners")
                bt.logging.debug("Sleeping for: 36 seconds")
                time.sleep(36)
                # If we receive empty responses from all axons, we can just set the scores to none for all the uids we queried
                for uid in list_of_uids:
                    bt.logging.trace(
                        f"Setting score for empty response from UID: {uid}. Old score: {validator.scores[uid]}"
                    )
                    validator.scores[uid] = (
                        validator.neuron_config.alpha * validator.scores[uid]
                        + (1 - validator.neuron_config.alpha) * 0.0
                    )
                    bt.logging.trace(
                        f"Set score for empty response from UID: {uid}. New score: {validator.scores[uid]}"
                    )
                continue

            bt.logging.trace(f"Received responses: {responses}")

            # Process the responses
            # processed_uids = torch.nonzero(list_of_uids).squeeze()
            response_data = validator.process_responses(
                processed_uids=list_of_uids,
                responses=responses
            )

            for res in response_data:
                if validator.miner_responses:
                    if res["hotkey"] in validator.miner_responses:
                        validator.miner_responses[res["hotkey"]].append(res)
                    else:
                        validator.miner_responses[res["hotkey"]] = [res]
                else:
                    validator.miner_responses = {}
                    validator.miner_responses[res["hotkey"]] = [res]
            

            if current_block - validator.last_updated_block > 200:
                # Periodically update the weights on the Bittensor blockchain.
                try:
                    validator.set_weights()
                    # Update validators knowledge of the last updated block
                    validator.last_updated_block = validator.subtensor.block
                except TimeoutError as e:
                    bt.logging.error(f"Setting weights timed out: {e}")
                # update local games database
                update_games

            # End the current step and prepare for the next iteration.
            validator.step += 1

            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            bt.logging.debug("Sleeping for: 18 seconds")
            time.sleep(18)
        except TimeoutError as e:
            bt.logging.debug("Validator timed out")
    

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--netuid", type=int, default=34, help="The chain subnet uid.")

    parser.add_argument(
        "--max_targets",
        type=int,
        default=128,
        help="Sets the value for the number of targets to query at once",
    )
    parser.add_argument(
        "--load_state",
        type=str,
        default="True",
        help="WARNING: Setting this value to False clears the old state.",
    )
    validator = BettensorValidator(parser=parser)

    if (
        not validator.apply_config(
            bt_classes=[bt.subtensor, bt.logging, bt.wallet]
        )
        or not validator.initialize_neuron()
    ):
        bt.logging.error("Unable to initialize Validator. Exiting.")
        sys.exit()

    main(validator)
