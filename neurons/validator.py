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
import asyncio
import threading

import bettensor
from bettensor import protocol
from bettensor.protocol import GameData, Metadata
from bettensor.validator.bettensor_validator import BettensorValidator
from bettensor.validator.utils.io.sports_data import SportsData
from bettensor.validator.utils.scoring.watchdog import Watchdog
from bettensor.validator.utils.io.auto_updater import *



async def async_operations(validator):
    # Create semaphores for each operation
    update_semaphore = asyncio.Semaphore(1)
    game_data_semaphore = asyncio.Semaphore(1)
    metagraph_semaphore = asyncio.Semaphore(1)
    query_semaphore = asyncio.Semaphore(1)
    website_semaphore = asyncio.Semaphore(1)
    scoring_semaphore = asyncio.Semaphore(1)
    weights_semaphore = asyncio.Semaphore(1)

    while True:
        current_time = datetime.now(timezone.utc)
        current_block = validator.subtensor.block

        # Perform update (if needed)
        if not update_semaphore.locked():
            asyncio.create_task(perform_update_task(validator, update_semaphore))

        # Update game data
        if (current_block - validator.last_updated_block) > validator.update_game_data_interval and not game_data_semaphore.locked():
            asyncio.create_task(update_game_data_task(validator, current_time, game_data_semaphore))

        # Sync metagraph
        if not metagraph_semaphore.locked():
            asyncio.create_task(sync_metagraph_task(validator, metagraph_semaphore))

        # Query and process axons
        if (current_block - validator.last_queried_block) > validator.query_axons_interval and not query_semaphore.locked():
            asyncio.create_task(query_and_process_axons_task(validator, query_semaphore))

        # Send data to website
        if (current_block - validator.last_sent_data_to_website) > validator.send_data_to_website_interval and not website_semaphore.locked():
            asyncio.create_task(send_data_to_website_task(validator, website_semaphore))

        # Recalculate scores
        if (current_block - validator.last_scoring_block) > validator.scoring_interval and not scoring_semaphore.locked():
            asyncio.create_task(scoring_run_task(validator, current_time, scoring_semaphore))

        # Set weights
        if (current_block - validator.last_set_weights_block) > validator.set_weights_interval and not weights_semaphore.locked():
            asyncio.create_task(set_weights_task(validator, weights_semaphore))

        await asyncio.sleep(12)  # Wait before next iteration

async def perform_update_task(validator, semaphore):
    async with semaphore:
        await perform_update(validator)

async def update_game_data_task(validator, current_time, semaphore):
    async with semaphore:
        await asyncio.to_thread(update_game_data, validator, current_time)

async def sync_metagraph_task(validator, semaphore):
    async with semaphore:
        await asyncio.to_thread(sync_metagraph, validator)

async def query_and_process_axons_task(validator, semaphore):
    async with semaphore:
        await asyncio.to_thread(query_and_process_axons_with_game_data, validator)

async def send_data_to_website_task(validator, semaphore):
    async with semaphore:
        await asyncio.to_thread(send_data_to_website_server, validator)

async def scoring_run_task(validator, current_time, semaphore):
    async with semaphore:
        await asyncio.to_thread(scoring_run, validator, current_time)

async def set_weights_task(validator, semaphore):
    async with semaphore:
        await set_weights(validator, validator.scores)




def main(validator: BettensorValidator):
    initialize(validator)
    watchdog = Watchdog(timeout=900)  # 15 minutes timeout

    # Create and run the async loop in a separate thread
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=async_loop.run_forever)
    async_thread.start()

    # Schedule the async operations
    asyncio.run_coroutine_threadsafe(async_operations(validator), async_loop)

    while True:
        try:
            watchdog.reset()
            current_block = validator.subtensor.block
            
            # Define default intervals if they don't exist
            if not hasattr(validator, 'update_game_data_interval'):
                validator.update_game_data_interval = 50  # Default value, adjust as needed

            if not hasattr(validator, 'query_axons_interval'):
                validator.query_axons_interval = 10  # Default value, adjust as needed

            if not hasattr(validator, 'send_data_to_website_interval'):
                validator.send_data_to_website_interval = 15  # Default value, adjust as needed

            if not hasattr(validator, 'scoring_interval'):
                validator.scoring_interval = 50  # Default value, adjust as needed

            if not hasattr(validator, 'set_weights_interval'):
                validator.set_weights_interval = 300  # Default value, adjust as needed

            # Define last operation block numbers if they don't exist
            if not hasattr(validator, 'last_queried_block'):
                validator.last_queried_block = validator.subtensor.block - 10

            if not hasattr(validator, 'last_sent_data_to_website'):
                validator.last_sent_data_to_website = validator.subtensor.block - 15

            if not hasattr(validator, 'last_scoring_block'):
                validator.last_scoring_block = validator.subtensor.block - 50

            if not hasattr(validator, 'last_set_weights_block'):
                validator.last_set_weights_block = validator.subtensor.block - 300

            # Calculate interval values
            current_block = validator.subtensor.get_current_block()
            blocks_until_query_axons = max(0, validator.query_axons_interval - ((current_block - validator.last_queried_block) % validator.query_axons_interval))
            blocks_until_send_data = max(0, validator.send_data_to_website_interval - ((current_block - validator.last_sent_data_to_website) % validator.send_data_to_website_interval))
            blocks_until_scoring = max(0, validator.scoring_interval - ((current_block - validator.last_scoring_block) % validator.scoring_interval))
            blocks_until_set_weights = max(0, validator.set_weights_interval - ((current_block - validator.last_set_weights_block) % validator.set_weights_interval))

            # Main loop now only handles watchdog reset and logging
            bt.logging.info("\n"
                "--------------------------------Status--------------------------------"
                "\n"
                f"Current Step: {validator.step},\n" 
                f"Current block: {current_block},\n" 
                f"Last updated block: {validator.last_updated_block},\n" 
                f"Blocks until next query_and_process_axons: {blocks_until_query_axons},\n" 
                f"Blocks until send_data_to_website: {blocks_until_send_data},\n" 
                f"Blocks until scoring_run: {blocks_until_scoring},\n" 
                f"Blocks until set_weights: {blocks_until_set_weights}"
                "\n"
                "----------------------------------------------------------------------"
            )
            
            validator.step += 1
            time.sleep(30)

        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt received. Shutting down gracefully...")
            break
        except TimeoutError as e:
            bt.logging.error(f"Error in main loop: {str(e)}")
            validator.initialize_connection()
        except Exception as e:
            bt.logging.error(f"Unexpected error in main loop: {str(e)}")
            bt.logging.error(traceback.format_exc())
            time.sleep(10)  # Wait before retrying

    # Cleanup
    async_loop.call_soon_threadsafe(async_loop.stop)
    async_thread.join()
    bt.logging.info("Validator shutdown complete.")


def initialize(validator):
    validator.serve_axon()
    validator.initialize_connection()

    if args.load_state.lower() == "true":
        validator.load_state()

    if not validator.last_updated_block:
        bt.logging.info("Updating last updated block; will set weights this iteration")
        validator.last_updated_block = validator.subtensor.block - 301
        validator.last_queried_block = validator.subtensor.block - 10
        validator.last_sent_data_to_website = validator.subtensor.block - 15
        validator.last_scoring_block = validator.subtensor.block - 50
        validator.last_set_weights_block = validator.subtensor.block - 300
        


def update_game_data(validator, current_time):
    """
    Calls SportsData to update game data in the database - Async in separate thread
    
    """
    bt.logging.info("--------------------------------Updating game data--------------------------------")
    
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

    validator.last_updated_block = validator.subtensor.block
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
    """
    Queries axons with game data and processes the responses
    """
    bt.logging.info("--------------------------------Querying and processing axons with game data--------------------------------")
    
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
            deserialize=False,
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
    bt.logging.info("Finished querying axons..")

    if not responses:
        print("No responses received. Sleeping for 18 seconds.")
        time.sleep(18)

    bt.logging.info(f"Received {len(responses)} responses")

    valid_responses = []
    invalid_responses = []

    #print one whole response for debugging
    #bt.logging.info(f"Received response: {responses[0][0]}")

    bt.logging.info("Starting response processing...")
    
    for idx, response in enumerate(responses):
        try:
            bt.logging.info(f"Processing response: {idx}")
            if response.metadata.synapse_type == "prediction":
                valid_responses.append(response)
                bt.logging.info(f"Received valid response: {response.metadata.synapse_type}")
                bt.logging.trace(f"Response: {response}")
            else:
                invalid_responses.append(response)
                bt.logging.warning(f"Received invalid response: {response.metadata.synapse_type}")
        except Exception as e:
            bt.logging.error(f"Error processing response: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            bt.logging.warning(f"Response: {response}")
            continue
    

    bt.logging.warning(f"Received {len(invalid_responses)} invalid responses: {[response.metadata.synapse_type for response in invalid_responses]}")
    bt.logging.warning(f"Affected Miners: {[response.metadata.neuron_uid for response in invalid_responses]}")

    bt.logging.info(f"Received {len(valid_responses)} valid responses - processing...")
    if valid_responses and any(valid_responses):
        try:
            validator.process_prediction(
                processed_uids=list_of_uids, synapses=valid_responses
            )
        except Exception as e:
            bt.logging.error(f"Error processing predictions: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    validator.last_queried_block = validator.subtensor.block


def send_data_to_website_server(validator):
    """
    Sends data to the website server
    """
    bt.logging.info("--------------------------------Sending data to website server--------------------------------")
    validator.last_sent_data_to_website = validator.subtensor.block

    try:
        result = validator.website_handler.fetch_and_send_predictions()
        bt.logging.info(f"Result status: {result}")
        if result:
            bt.logging.info("Predictions fetched and sent successfully")
        else:
            bt.logging.warning("No new predictions were sent this round")
    except Exception as e:
        bt.logging.error(f"Error in fetch_and_send_predictions: {str(e)}")

    validator.last_sent_data_to_website = validator.subtensor.block


def scoring_run(validator, current_time):
    """
    calls the scoring system to update miner scores before setting weights
    """
    bt.logging.info("--------------------------------Scoring run--------------------------------")
    
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

    validator.last_scoring_block = validator.subtensor.block


async def set_weights(validator, scores):
    """
    Sets the weights for the validator
    """
    bt.logging.info("--------------------------------Setting weights--------------------------------")

    try:
        bt.logging.info("Attempting to update weights")
        if validator.subtensor is None:
            bt.logging.warning("Subtensor is None. Attempting to reinitialize...")
            validator.subtensor = validator.initialize_connection()

        if validator.subtensor is not None:
            success = await validator.set_weights(scores)
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

    validator.last_set_weights_block = validator.subtensor.block


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