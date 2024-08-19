# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 oneandahalfcats
# Copyright © 2023 geardici

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
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
from bettensor.utils.watchdog import Watchdog

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
            {"id": "2", "season": 2024},  # UEFA Champions League, World
            {"id": "3", "season": 2024},  # UEFA Europa League, World
            {"id": "848", "season": 2024},  # UEFA Europa Conference League, World
            {"id": "531", "season": 2024},  # UEFA Super Cup, World
            {"id": "140", "season": 2024},  # La Liga, Spain
            {"id": "78", "season": 2024},  # Bundesliga, Germany
            {"id": "135", "season": 2024},  # Serie A, Italy
            {"id": "39", "season": 2024},  # Premier League, England
            {"id": "61", "season": 2024},  # Ligue 1, France
            {"id": "94", "season": 2024},  # Primeira Liga, Portugal
            {"id": "71", "season": 2024},  # Serie A, Brazil
            {"id": "97", "season": 2024},  # Taça da Liga, Portugal
            {"id": "73", "season": 2024},  # Copa Do Brasil, Brazil
            {"id": "253", "season": 2024},  # Major League Soccer, USA
            {"id": "307", "season": 2024},  # Pro League, Saudi Arabia
            {"id": "262", "season": 2024},  # Liga MX, Mexico
            {"id": "12", "season": 2024},  # CAF Champions League, World
            {"id": "13", "season": 2024},  # CONMEBOL Libertadores, World
            {"id": "11", "season": 2024},  # CONMEBOL Sudamericana, World
            {"id": "81", "season": 2024},  # DFB Pokal, Germany
            {"id": "137", "season": 2024},  # Coppa Italia, Italy
            {"id": "526", "season": 2024},  # Trophée des Champions, France
            {"id": "529", "season": 2024},  # Super Cup, Germany
            {"id": "556", "season": 2024},  # Super Cup, Spain
            {"id": "103", "season": 2024},  # Eliteserien, Norway
            {"id": "98", "season": 2024},  # J1 League, Japan
            {"id": "274", "season": 2024},  # Liga 1, Indonesia
            {"id": "218", "season": 2024},  # Bundesliga, Austria
            {"id": "219", "season": 2024},  # 2. Liga, Austria
            {"id": "104", "season": 2024},  # 1. Division, Norway
            {"id": "107", "season": 2024},  # I Liga, Poland
            {"id": "88", "season": 2024},  # Eredivisie, Netherlands
            {"id": "540", "season": 2024},  # CONMEBOL Libertadores U20, World
            {"id": "383", "season": 2024},  # Ligat Ha'al, Israel
            {"id": "303", "season": 2024},  # Division 1, United-Arab-Emirates
            {"id": "304", "season": 2024},  # Liga Panameña de Fútbol, Panama
            {"id": "611", "season": 2024},  # Brasiliense, Brazil
            {"id": "77", "season": 2024},  # Alagoano, Brazil
            {"id": "602", "season": 2024},  # Baiano - 1, Brazil
            {"id": "624", "season": 2024},  # Carioca - 1, Brazil
            {"id": "604", "season": 2024},  # Catarinense - 1, Brazil
            {"id": "609", "season": 2024},  # Cearense - 1, Brazil
            {"id": "475", "season": 2024},  # Paulista - A1, Brazil
            {"id": "622", "season": 2024},  # Pernambucano - 1, Brazil
            {"id": "306", "season": 2024},  # Qatar Stars League, Qatar
            {"id": "19", "season": 2024},  # AFC Champions League, Asia
            {"id": "101", "season": 2024},  # J-League Cup, Japan
            {"id": "72", "season": 2024},  # Serie B, Brazil
            {"id": "136", "season": 2024},  # Serie B, Italy
            {"id": "812", "season": 2024},  # Super Cup, Belarus
        ]
    }
    # commenting this out to prevent excessive api calls if validator crashes - strictly once/hour now
    """ try:
        all_games = await validator.run_sync_in_async(lambda: sports_data.get_multiple_game_data(sports_config))
        if all_games is None:
            bt.logging.warning("Failed to fetch game data. Continuing with previous data.")
        else:
            validator.last_api_call = datetime.now()
    except Exception as e:
        bt.logging.error(f"Error fetching game data: {e}")
        # Continue with the previous data
 """
    validator.serve_axon()
    await validator.initialize_connection()

    watchdog = Watchdog(timeout=300)  # 5 minutes timeout

    bt.logging.info("Recalculating daily profits...")
    validator.recalculate_all_profits() # Running this at startup, then excluding it from the loop

    while True:

        try:
            watchdog.reset()
            current_time = datetime.now(timezone.utc)
            
            # Ensure last_api_call is a datetime object
            if not isinstance(validator.last_api_call, datetime):
                validator.last_api_call = datetime.fromtimestamp(validator.last_api_call, tz=timezone.utc)
            
            # Update games every hour
            if current_time - validator.last_api_call >= timedelta(hours=1):
                try:
                    all_games = await validator.run_sync_in_async(lambda: sports_data.get_multiple_game_data(sports_config))
                    if all_games is None:
                        bt.logging.warning("Failed to fetch game data. Continuing with previous data.")
                    else:
                        validator.last_api_call = current_time
                        validator.save_state()  # Save state after updating last_api_call
                except Exception as e:
                    bt.logging.error(f"Error fetching game data: {e}")
                    # Continue with the previous data

            # Ensure last_update_recent_games is a datetime object
            if not isinstance(validator.last_update_recent_games, datetime):
                validator.last_update_recent_games = datetime.fromtimestamp(validator.last_update_recent_games, tz=timezone.utc)

            # Update recent games every 30 minutes
            if current_time - validator.last_update_recent_games >= timedelta(minutes=30):
                try:
                    await validator.run_sync_in_async(validator.update_recent_games)
                    validator.last_update_recent_games = current_time
                    validator.save_state()  # Save state after updating last_update_recent_games
                except Exception as e:
                    bt.logging.error(f"Error updating recent games: {str(e)}")

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
            if validator.scores is None:
                bt.logging.warning("Scores were None. Reinitializing...")
                validator.init_default_scores()
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


            # Batch query the neurons to avoid timeout errors, 20 at a time
            responses = []

            for i in range(0, len(uids_to_query), 20):
                responses += validator.dendrite.query(
                    axons=uids_to_query[i:i+20],
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

            if current_block - validator.last_updated_block > 30:
                # Sends data to the website
                try:
                    result = fetch_and_send_predictions("data/validator.db")
                    bt.logging.info(f"Result status: {result}")
                    if result:
                        bt.logging.info("Predictions fetched and sent successfully")
                    else:
                        bt.logging.warning("No new predictions were sent this round")
                except Exception as e:
                    bt.logging.error(f"Error in fetch_and_send_predictions: {str(e)}")

            current_time = datetime.now(timezone.utc)
            if current_time - validator.last_api_call >= timedelta(minutes=30):  # 30 minutes in seconds
                # Update results before setting weights next block
                await validator.run_sync_in_async(validator.update_recent_games)
                validator.last_api_call = current_time
                validator.save_state()  # Save state after updating last_api_call
            
            if current_block - validator.last_updated_block > 300:

                try:
                    bt.logging.info("Attempting to update weights")
                    if validator.subtensor is None:
                        bt.logging.warning("Subtensor is None. Attempting to reinitialize...")
                        validator.subtensor = await validator.initialize_connection()
                    
                    if validator.subtensor is not None:
                        success = await validator.set_weights()
                        bt.logging.info("Weight update attempt completed")
                    else:
                        bt.logging.error("Failed to reinitialize subtensor. Skipping weight update.")
                        success = False
                except Exception as e:
                    bt.logging.error(f"Error during weight update process: {str(e)}")
                    success = False

                # Update last_updated_block regardless of the outcome
                try:
                    validator.last_updated_block = await validator.run_sync_in_async(lambda: validator.subtensor.block)
                    bt.logging.info(f"Updated last_updated_block to {validator.last_updated_block}")
                except Exception as e:
                    bt.logging.error(f"Error updating last_updated_block: {str(e)}")

                if success:
                    bt.logging.info("Successfully updated weights")
                else:
                    bt.logging.warning("Failed to set weights or encountered an error, continuing with next iteration.")

            # End the current step and prepare for the next iteration.
            validator.step += 1
            watchdog.reset()
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            bt.logging.debug("Sleeping for: 45 seconds")
            await asyncio.sleep(45)

            #bt.logging.warning(f"TESTING AUTO UPDATE!!")

        except TimeoutError as e:
            bt.logging.error(f"Error in main loop: {str(e)}")
            # Attempt to reconnect if necessary
            await validator.initialize_connection()


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
        default=256,
        help="Sets the value for the number of targets to query - set to 256 to ensure all miners are querie, it is now batched",
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
