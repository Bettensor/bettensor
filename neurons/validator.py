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
import torch
import time
import sys
import os


from bettensor.protocol import GameData
from bettensor.protocol import GameData, Metadata
from bettensor.utils.sports_data import SportsData

# Bittensor
import bittensor as bt
import copy
from copy import deepcopy

# Bittensor Validator Template:
import bettensor
from uuid import UUID
from argparse import ArgumentParser
import sqlite3

# need to import the right protocol(s) here
from bettensor.validator.bettensor_validator import BettensorValidator
from bettensor import protocol

# from update_games import update_games
from bettensor.utils.miner_stats import MinerStatsHandler
from datetime import datetime, timezone
from bettensor.utils.website_handler import fetch_and_send_predictions

def main(validator: BettensorValidator):
    sports_data = SportsData()
    baseball_games = sports_data.get_game_data(
        sport="baseball", league="1", season="2024"
    )
    soccer_games = sports_data.get_game_data(
        sport="soccer", league="253", season="2024"
    )  # MLS
    soccer_games = sports_data.get_game_data(
        sport="soccer", league="140", season="2024"
    )  # La Liga
    soccer_games = sports_data.get_game_data(
        sport="soccer", league="78", season="2024"
    )  # Bundesliga
    soccer_games = sports_data.get_game_data(
        sport="soccer", league="262", season="2024"
    )  # Liga MX

    while True:
        try:
            # Periodically sync subtensor status and save the state file
            if validator.step % 5 == 0:
                # Sync metagraph
                try:
                    validator.metagraph = validator.sync_metagraph(
                        validator.metagraph, validator.subtensor
                    )
                    bt.logging.debug(f"Metagraph synced: {validator.metagraph}")
                except TimeoutError as e:
                    bt.logging.error(f"Metagraph sync timed out: {e}")

                # Update local knowledge of the hotkeys
                validator.check_hotkeys()

                # Save state
                validator.save_state()

            if validator.step % 20 == 0:
                pass
                # Update local knowledge of blacklisted miner hotkeys
                # validator.check_blacklisted_miner_hotkeys()
            
            if validator.step % 10 == 0:
                result = fetch_and_send_predictions(db_path="data/validator.db")
                bt.logging.trace(f"result status: {result}")
                if result:
                    bt.logging.debug(
                        "Predictions fetched and sent successfully:", result
                    )
                else:
                    bt.logging.debug("Failed to fetch or send predictions")
            # Get all axons
            all_axons = validator.metagraph.axons
            bt.logging.trace(f"All axons: {all_axons}")

            # If there are more axons than scores, append the scores list and add new miners to the database
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

            validator.add_new_miners()

            # Get list of UIDs to query
            (
                uids_to_query,
                list_of_uids,
                blacklisted_uids,
                uids_not_to_query,
            ) = validator.get_uids_to_query(all_axons=all_axons)
            if not uids_to_query:
                bt.logging.warning(f"UIDs to query is empty: {uids_to_query}")

            # Get data and populate DB
            if validator.step % 20 == 0:
                baseball_games = sports_data.get_game_data(
                    sport="baseball", league="1", season="2024"
                )
                soccer_games = sports_data.get_game_data(
                    sport="soccer", league="253", season="2024"
                )
                validator.set_weights()
            # Broadcast query to valid Axons
            current_time = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            # metadata = Metadata.create(validator.wallet, validator.subnet_version, validator.uid)
            print(f"uids_to_query: {uids_to_query}")

            synapse = GameData.create(
                db_path=validator.db_path,
                wallet=validator.wallet,
                subnet_version=validator.subnet_version,
                neuron_uid=validator.uid,
                synapse_type="game_data",
            )
            bt.logging.info(
                f"Synapse: {synapse.metadata.synapse_id} , {synapse.metadata.timestamp}, type: {synapse.metadata.synapse_type}, origin: {synapse.metadata.neuron_uid}"
            )
            responses = validator.dendrite.query(
                axons=uids_to_query,
                synapse=synapse,
                timeout=validator.timeout,
                deserialize=True,
            )

            # Process blacklisted UIDs (set scores to 0)
            bt.logging.debug(f"blacklisted_uids: {blacklisted_uids}")
            for uid in blacklisted_uids:
                if uid is not None:
                    bt.logging.debug(
                        f"Setting score for blacklisted UID: {uid}. Old score: {validator.scores[uid]}"
                    )
                    validator.scores[uid] = (
                        validator.neuron_config.alpha * validator.scores[uid]
                        + (1 - validator.neuron_config.alpha) * 0.0
                    )
                    bt.logging.debug(
                        f"Set score for blacklisted UID: {uid}. New score: {validator.scores[uid]}"
                    )

            # Process UIDs we did not query (set scores to 0)
            bt.logging.debug(f"uids_not_to_query: {uids_not_to_query}")
            for uid in uids_not_to_query:
                if uid is not None:
                    bt.logging.trace(
                        f"Setting score for not queried UID: {uid}. Old score: {validator.scores[uid]}"
                    )

                    validator_alpha_type = type(validator.neuron_config.alpha)
                    validator_scores_type = type(validator.scores[uid])

                    bt.logging.debug(
                        f"validator_alpha_type: {validator_alpha_type}, validator_scores_type: {validator_scores_type}"
                    )
                    validator.scores[uid] = (
                        validator.neuron_config.alpha * validator.scores[uid]
                        + (1 - validator.neuron_config.alpha) * 0.0
                    )
                    bt.logging.trace(
                        f"Set score for not queried UID: {uid}. New score: {validator.scores[uid]}"
                    )
            if not responses:
                print("No responses received. Sleeping for 18 seconds.")
                time.sleep(18)
            # Log the results for monitoring purposes.
            # elif all(item.output is None for item in responses):
            #    bt.logging.info("Received empty response from all miners")
            #    bt.logging.debug("Sleeping for: 36 seconds")
            #    time.sleep(18)
            # If we receive empty responses from all axons, we can just set the scores to none for all the uids we queried
            #    for uid in list_of_uids:
            #        bt.logging.trace(
            #            f"Setting score for empty response from UID: {uid}. Old score: {validator.scores[uid]}"
            #        )
            #        validator.scores[uid] = (
            #            validator.neuron_config.alpha * validator.scores[uid]
            #            + (1 - validator.neuron_config.alpha) * 0.0
            #        )
            #        bt.logging.trace(
            #            f"Set score for empty response from UID: {uid}. New score: {validator.scores[uid]}"
            #        )
            #    continue

            bt.logging.trace(f"Received responses: {responses}")

            # Process the responses
            # processed_uids = torch.nonzero(list_of_uids).squeeze()
            if responses and any(responses):
                validator.process_prediction(
                    processed_uids=list_of_uids, predictions=responses
                )

            current_block = validator.subtensor.block

            bt.logging.debug(
                f"Current Step: {validator.step}, Current block: {current_block}, last_updated_block: {validator.last_updated_block}"
            )

            if current_block - validator.last_updated_block > 200:
                # Periodically update the weights on the Bittensor blockchain.
                try:
                    validator.set_weights()
                    # Update validators knowledge of the last updated block
                    validator.last_updated_block = validator.subtensor.block
                except TimeoutError as e:
                    bt.logging.error(f"Setting weights timed out: {e}")
                # update local games database
                # update_games()

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

    parser.add_argument(
        "--alpha", type=float, default=0.9, help="The alpha value for the validator."
    )

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
        not validator.apply_config(bt_classes=[bt.subtensor, bt.logging, bt.wallet])
        or not validator.initialize_neuron()
    ):
        bt.logging.error("Unable to initialize Validator. Exiting.")
        sys.exit()

    main(validator)
