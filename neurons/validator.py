# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 oneandahalfcats
# Copyright © 2023 geardici

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
from dotenv import load_dotenv
import asyncio


# need to import the right protocol(s) here
from bettensor.validator.bettensor_validator import BettensorValidator
from bettensor import protocol

# from update_games import update_games
from bettensor.utils.miner_stats import MinerStatsHandler
from datetime import datetime, timezone, timedelta
from bettensor.utils.website_handler import fetch_and_send_predictions

async def main(validator: BettensorValidator):
    # load rapid API key
    load_dotenv()
    rapid_api_key = os.getenv('RAPID_API_KEY')

    sports_data = SportsData()
    sports_config = {
        "baseball": [
            {"id": "1", "season": "2024"}
        ],
        "soccer": [
            {"id": "253", "season": "2024"},  # MLS
            {"id": "140", "season": "2024"},  # La Liga
            {"id": "78", "season": "2024"},   # Bundesliga
            {"id": "262", "season": "2024"},  # Liga MX
            {"id": "4", "season": "2024"},     # Euro Cup
            {"id": "9", "season": "2024"},    # Copa America
            {"id": "71", "season": "2024"},  # Brasileirão Série A
            {"id": "98", "season": "2024"}, # J1 League
            {"id": "480", "season": "2024"}, # Olympics mens
            {"id": "524", "season": "2024"} # Olympics womens
        ]
    }

    all_games =  await validator.run_sync_in_async(lambda: sports_data.get_multiple_game_data(sports_config))
    last_api_call = datetime.now()

    validator.serve_axon()
    await validator.initialize_connection()

    while True:

        try:
            current_time = datetime.now()
            if current_time - last_api_call >= timedelta(hours=1):
                # Update games every hour
                all_games = await validator.run_sync_in_async(lambda: sports_data.get_multiple_game_data(sports_config))
                last_api_call = current_time

            # Periodically sync subtensor status and save the state file
            if validator.step % 5 == 0:
                # Sync metagraph
                try:
                    validator.metagraph = await validator.sync_metagraph()
                    bt.logging.debug(f"Metagraph synced: {validator.metagraph}")
                except TimeoutError as e:
                    bt.logging.error(f"Metagraph sync timed out: {e}")

                # Update local knowledge of the hotkeys
                validator.check_hotkeys()

                # Save state
                validator.save_state()

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

            # Broadcast query to valid Axons
            current_time = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            # metadata = Metadata.create(validator.wallet, validator.subnet_version, validator.uid)

            synapse = GameData.create(
                db_path=validator.db_path,
                wallet=validator.wallet,
                subnet_version=validator.subnet_version,
                neuron_uid=validator.uid,
                synapse_type="game_data",
            )
            bt.logging.debug(
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

            # Process the responses
            if responses and any(responses):
                validator.process_prediction(
                    processed_uids=list_of_uids, predictions=responses
                )

            current_block = await validator.run_sync_in_async(lambda: validator.subtensor.block)

            bt.logging.debug(
                f"Current Step: {validator.step}, Current block: {current_block}, last_updated_block: {validator.last_updated_block}"
            )

            if current_block - validator.last_updated_block > 300:
                # Sends data to the website
                try:
                    result = fetch_and_send_predictions("data/validator.db")
                    bt.logging.info(f"Result status: {result}")
                    if result:
                        bt.logging.info("Predictions fetched and sent successfully")
                    else:
                        bt.logging.warning("No predictions were sent or an error occurred")
                except Exception as e:
                    bt.logging.error(f"Error in fetch_and_send_predictions: {str(e)}")

            if current_block - validator.last_updated_block > 299:
                # Update results before setting weights next block
                await validator.run_sync_in_async(validator.update_recent_games)
                
            if current_block - validator.last_updated_block > 300:

                try:
                    bt.logging.info("Attempting to update weights")
                    if validator.subtensor is None:
                        bt.logging.warning("Subtensor is None. Attempting to reinitialize...")
                        validator.subtensor = await validator.initialize_connection()
                    
                    if validator.subtensor is not None:
                        success = await validator.set_weights()
                        if success:
                            validator.last_updated_block = await validator.run_sync_in_async(lambda: validator.subtensor.block)
                            bt.logging.info("Successfully updated weights and last updated block")
                        else:
                            bt.logging.warning("Failed to set weights, continuing with next iteration.")
                    else:
                        bt.logging.error("Failed to reinitialize subtensor. Skipping weight update.")
                except Exception as e:
                    bt.logging.error(f"Error during weight update process: {str(e)}")
                    bt.logging.warning("Continuing with next iteration despite weight update failure.")

            # End the current step and prepare for the next iteration.
            validator.step += 1

            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            bt.logging.debug("Sleeping for: 18 seconds")
            await asyncio.sleep(18)

            #bt.logging.warning(f"TESTING AUTO UPDATE!!")

        except TimeoutError as e:
            bt.logging.error(f"Error in main loop: {str(e)}")
            # Attempt to reconnect if necessary
            await self.initialize_connection()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--subtensor.network', type=str, help="The subtensor network to connect to")
    parser.add_argument('--subtensor.chain_endpoint', type=str, help="The subtensor network to connect to")
    parser.add_argument('--wallet.name', type=str, help="The name of the wallet to use")
    parser.add_argument('--wallet.hotkey', type=str, help="The hotkey of the wallet to use")
    parser.add_argument('--logging.trace', action='store_true', help="Enable trace logging")
    parser.add_argument(
        "--alpha", type=float, default=0.9, help="The alpha value for the validator."
    )
    parser.add_argument("--netuid", type=int, default=30, help="The chain subnet uid.")
    parser.add_argument('--axon.port', type=int, help="The port this axon endpoint is serving on.")
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
    args = parser.parse_args()
    print("Parsed arguments:", args)
    validator = BettensorValidator(parser=parser)

    if (
        not validator.apply_config(bt_classes=[bt.subtensor, bt.logging, bt.wallet])
        or not validator.initialize_neuron()
    ):
        bt.logging.error("Unable to initialize Validator. Exiting.")
        sys.exit()

    asyncio.get_event_loop().run_until_complete(main(validator))
