import os
import sys
import time
import copy
import traceback
import torch
import sqlite3
import bittensor as bt
from uuid import UUID
from dotenv import load_dotenv
from copy import deepcopy
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

import bettensor
from bettensor import protocol
from bettensor.protocol import GameData, Metadata
from bettensor.validator.bettensor_validator import BettensorValidator
from bettensor.validator.utils.io.sports_data import SportsData
from bettensor.validator.utils.scoring.watchdog import Watchdog
from bettensor.validator.utils.io.website_handler import fetch_and_send_predictions


def main(validator: BettensorValidator):
    initialize(validator)
    watchdog = Watchdog(timeout=900)  # 15 minutes timeout

    while True:
        try:
            # reset watchdog
            watchdog.reset()

            # get current time and block
            current_time = datetime.now(timezone.utc)
            current_block = validator.subtensor.block

            # Convert last_api_call to datetime if it's a string
            if isinstance(validator.last_api_call, str):
                validator.last_api_call = datetime.fromisoformat(
                    validator.last_api_call
                )

            bt.logging.info("Current time(main loop): ", current_time)
            bt.logging.info("Last api call was at: ", validator.last_api_call)

            # Update game data every 10 minutes with bettensor api
            if current_time - validator.last_api_call >= timedelta(minutes=10):
                bt.logging.info("Updating game data with bettensor api")
                update_game_data(validator, current_time)
            # Sync metagraph every step
            sync_metagraph(validator)

            # Query and process axons with game data every 10 blocks (~2 minutes)
            if (current_block - validator.last_updated_block) % 10 > 0:
                query_and_process_axons_with_game_data(validator)

            # Send data to website server every 15 blocks (~ 3 minutes)
            if (current_block - validator.last_updated_block) % 15 > 0:
                send_data_to_website_server(validator)

            # Recalculate scores every 50 blocks (~2 minutes)
            if current_block - validator.last_updated_block > 50:
                bt.logging.info("Recalculating scores")
                scoring_run(validator, current_time)

            # Set weights every 300 blocks (~ 1 hour)
            if current_block - validator.last_updated_block > 300:
                set_weights(validator, validator.scores)

            end_of_loop_processes(validator, watchdog, current_block)

        except TimeoutError as e:
            bt.logging.error(f"Error in main loop: {str(e)}")
            validator.initialize_connection()


def initialize(validator):
    validator.serve_axon()
    validator.initialize_connection()

    if args.load_state.lower() == "true":
        validator.load_state()

    if not validator.last_updated_block:
        bt.logging.info("Updating last updated block; will set weights this iteration")
        validator.last_updated_block = validator.subtensor.block - 301


def update_game_data(validator, current_time):
    """
    Calls SportsData to update game data in the database
    """
    try:
        if validator.last_api_call is None:
            validator.last_api_call = current_time - timedelta(days=15)

        all_games = validator.sports_data.fetch_and_update_game_data(
            validator.last_api_call
        )
        if all_games is None:
            bt.logging.warning(
                "Failed to fetch game data. Continuing with previous data."
            )

        bt.logging.info(f"Current time: {current_time}")
        validator.last_api_call = current_time

        bt.logging.info(f"Last api call updated to: {validator.last_api_call}")
        validator.save_state()

    except Exception as e:
        bt.logging.error(f"Error fetching game data: {e}")
        bt.logging.error(f"Traceback:\n{traceback.format_exc()}")


def sync_metagraph(validator):
    try:
        validator.metagraph = validator.sync_metagraph()
        bt.logging.debug(f"Metagraph synced: {validator.metagraph}")
    except TimeoutError as e:
        bt.logging.error(f"Metagraph sync timed out: {e}")

    validator.check_hotkeys()
    validator.save_state()


def filter_and_update_axons(validator):
    all_axons = validator.metagraph.axons
    bt.logging.trace(f"All axons: {all_axons}")

    if validator.scores is None:
        bt.logging.warning("Scores were None. Reinitializing...")
        validator.init_default_scores()

    if validator.scores is None:
        bt.logging.error("Failed to initialize scores. Exiting.")
        return None, None, None, None

    num_uids = len(validator.metagraph.uids.tolist())
    current_scores_len = len(validator.scores)

    if num_uids > current_scores_len:
        bt.logging.info(f"Discovered new Axons, current scores: {validator.scores}")
        validator.scores = torch.cat(
            (
                validator.scores,
                torch.zeros(
                    (num_uids - current_scores_len),
                    dtype=torch.float32,
                ),
            )
        )
        bt.logging.info(f"Updated scores, new scores: {validator.scores}")

    (
        uids_to_query,
        list_of_uids,
        blacklisted_uids,
        uids_not_to_query,
    ) = validator.get_uids_to_query(all_axons=all_axons)

    if not uids_to_query:
        bt.logging.warning(f"UIDs to query is empty: {uids_to_query}")

    return uids_to_query, list_of_uids, blacklisted_uids, uids_not_to_query


def query_and_process_axons_with_game_data(validator):
    current_time = datetime.now(timezone.utc).isoformat()
    gamedata_dict = validator.fetch_local_game_data(current_timestamp=current_time)
    if gamedata_dict is None:
        bt.logging.error("No game data found")
        return None

    synapse = GameData.create(
        db_path=validator.db_path,
        wallet=validator.wallet,
        subnet_version=validator.subnet_version,
        neuron_uid=validator.uid,
        synapse_type="game_data",
        gamedata_dict=gamedata_dict,
    )
    if synapse is None:
        bt.logging.error("Synapse is None")
        return None

    bt.logging.debug(
        f"Synapse: {synapse.metadata.synapse_id} , {synapse.metadata.timestamp}, type: {synapse.metadata.synapse_type}, origin: {synapse.metadata.neuron_uid}"
    )

    responses = []
    result = filter_and_update_axons(validator)
    if result is None:
        bt.logging.error("Failed to filter and update axons")
        return None

    uids_to_query, list_of_uids, blacklisted_uids, uids_not_to_query = result

    for i in range(0, len(uids_to_query), 20):
        responses += validator.dendrite.query(
            axons=uids_to_query[i : i + 20],
            synapse=synapse,
            timeout=validator.timeout,
            deserialize=True,
        )

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

        if responses and any(responses):
            validator.process_prediction(
                processed_uids=list_of_uids, predictions=responses
            )

    return synapse


def send_data_to_website_server(validator):
    current_block = validator.subtensor.block

    try:
        result = fetch_and_send_predictions("data/validator.db")
        bt.logging.info(f"Result status: {result}")
        if result:
            bt.logging.info("Predictions fetched and sent successfully")
        else:
            bt.logging.warning("No new predictions were sent this round")
    except Exception as e:
        bt.logging.error(f"Error in fetch_and_send_predictions: {str(e)}")


def scoring_run(validator, current_time):
    """
    calls the scoring system to update miner scores before setting weights
    """
    try:
        # Get UIDs to query and invalid UIDs
        (
            _,
            list_of_uids,
            blacklisted_uids,
            uids_not_to_query,
        ) = validator.get_uids_to_query(validator.metagraph.axons)

        valid_uids = set(list_of_uids)
        # Combine blacklisted_uids and uids_not_to_query
        invalid_uids = set(blacklisted_uids + uids_not_to_query)

        

        validator.scores = validator.scoring_system.scoring_run(
            current_time, invalid_uids, valid_uids
        )
        bt.logging.info("Scores updated successfully")
        bt.logging.info(f"Scores: {validator.scores}")
    except Exception as e:
        bt.logging.error(f"Error in scoring_run: {str(e)}")
        bt.logging.error(f"Traceback: {traceback.format_exc()}")


def set_weights(validator, scores):
    try:
        bt.logging.info("Attempting to update weights")
        if validator.subtensor is None:
            bt.logging.warning("Subtensor is None. Attempting to reinitialize...")
            validator.subtensor = validator.initialize_connection()

        if validator.subtensor is not None:
            success = validator.set_weights(scores)
            bt.logging.info("Weight update attempt completed")
        else:
            bt.logging.error(
                "Failed to reinitialize subtensor. Skipping weight update."
            )
            success = False
    except Exception as e:
        bt.logging.error(f"Error during weight update process: {str(e)}")
        success = False

    try:
        validator.last_updated_block = validator.subtensor.block
        bt.logging.info(f"Updated last_updated_block to {validator.last_updated_block}")
    except Exception as e:
        bt.logging.error(f"Error updating last_updated_block: {str(e)}")

    if success:
        bt.logging.info("Successfully updated weights")
    else:
        bt.logging.warning(
            "Failed to set weights or encountered an error, continuing with next iteration."
        )


def end_of_loop_processes(validator, watchdog, current_block):
    validator.step += 1
    bt.logging.debug(
        f"Current Step: {validator.step}, Current block: {current_block}, last_updated_block: {validator.last_updated_block}"
    )
    watchdog.reset()
    bt.logging.debug("Sleeping for: 45 seconds")
    time.sleep(45)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--subtensor.network", type=str, help="The subtensor network to connect to"
    )
    parser.add_argument(
        "--subtensor.chain_endpoint",
        type=str,
        help="The subtensor network to connect to",
    )
    parser.add_argument("--wallet.name", type=str, help="The name of the wallet to use")
    parser.add_argument(
        "--wallet.hotkey", type=str, help="The hotkey of the wallet to use"
    )
    parser.add_argument(
        "--logging.trace", action="store_true", help="Enable trace logging"
    )
    parser.add_argument(
        "--logging.debug", action="store_true", help="Enable debug logging"
    )

    parser.add_argument(
        "--alpha", type=float, default=0.9, help="The alpha value for the validator."
    )
    parser.add_argument("--netuid", type=int, default=30, help="The chain subnet uid.")
    parser.add_argument(
        "--axon.port", type=int, help="The port this axon endpoint is serving on."
    )
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

    main(validator)
