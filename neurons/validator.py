import os
import subprocess
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
import websocket
from websocket._exceptions import WebSocketConnectionClosedException
from bettensor.protocol import GameData, Metadata
from bettensor.validator.bettensor_validator import BettensorValidator
from bettensor.validator.utils.ensure_dependencies import ensure_dependencies
from bettensor.validator.utils.io.sports_data import SportsData
from bettensor.validator.utils.scoring.watchdog import Watchdog
from bettensor.validator.utils.io.auto_updater import perform_update
import threading
import asyncio
from functools import partial
from typing import Optional, Any
import async_timeout
from bettensor.validator.utils.state_sync import StateSync
import math
from bettensor.validator.utils.database.database_manager import DatabaseManager

# Constants for timeouts (in seconds)
UPDATE_TIMEOUT = 300  # 5 minutes
GAME_DATA_TIMEOUT = 180  # 3 minutes
METAGRAPH_TIMEOUT = 120  # 2 minutes
QUERY_TIMEOUT = 600  # 10 minutes
WEBSITE_TIMEOUT = 60  # 1 minute
SCORING_TIMEOUT = 300  # 5 minutes
WEIGHTS_TIMEOUT = 180  # 3 minutes



def time_task(task_name):
    """
    Decorator that times the execution of validator tasks and logs the duration.
    Handles both sync and async functions.
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                bt.logging.info(f"{task_name} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                duration = time.time() - start_time
                bt.logging.error(f"{task_name} failed after {duration:.2f} seconds with error: {str(e)}")
                raise

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                bt.logging.info(f"{task_name} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                duration = time.time() - start_time
                bt.logging.error(f"{task_name} failed after {duration:.2f} seconds with error: {str(e)}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def log_status(validator):
    while True:
        current_time = datetime.now(timezone.utc)
        current_block = validator.subtensor.block
        blocks_until_query_axons = max(0, validator.query_axons_interval - (current_block - validator.last_queried_block))
        blocks_until_send_data = max(0, validator.send_data_to_website_interval - (current_block - validator.last_sent_data_to_website))
        blocks_until_scoring = max(0, validator.scoring_interval - (current_block - validator.last_scoring_block))
        blocks_until_set_weights = max(0, validator.set_weights_interval - (current_block - validator.last_set_weights_block))

        status_message = (
            "\n"
            "================================ VALIDATOR STATUS ================================\n"
            f"Current time: {current_time}\n"
            f"Scoring System, Current Day: {validator.scoring_system.current_day}\n"
            f"Current block: {current_block}\n"
            f"Last updated block: {validator.last_updated_block}\n"
            f"Blocks until next query_and_process_axons: {blocks_until_query_axons}\n"
            f"Blocks until send_data_to_website: {blocks_until_send_data}\n"
            f"Blocks until scoring_run: {blocks_until_scoring}\n"
            f"Blocks until set_weights: {blocks_until_set_weights}\n"
            "================================================================================\n"
        )
        
        debug_message = (
            f"Scoring System, Current Day: {validator.scoring_system.current_day}\n"
            f"Scoring System, Current Day Tiers: {validator.scoring_system.tiers[:, validator.scoring_system.current_day]}\n"
            f"Scoring System, Current Day Tiers Length: {len(validator.scoring_system.tiers[:, validator.scoring_system.current_day])}\n"
            f"Scoring System, Current Day Scores: {validator.scoring_system.composite_scores[:, validator.scoring_system.current_day, 0]}\n"
            f"Scoring System, Amount Wagered Last 5 Days: {validator.scoring_system.amount_wagered[:, validator.scoring_system.current_day]}\n"

        )

        bt.logging.info(status_message)
        #bt.logging.debug(debug_message)
        time.sleep(30)

def game_data_update_loop(validator):
    """Background thread function for updating game data"""
    while True:
        try:
            current_time = datetime.now(timezone.utc)
            bt.logging.debug("Attempting to run game data update...")
            
            # Try to acquire the lock without blocking
            if validator.operation_lock.locked():
                bt.logging.debug("Skipping game data update - query_and_process_axons is running")
                time.sleep(30)
                continue
                
            # Use asyncio.run to handle the async lock in a sync context
            asyncio.run(update_game_data_with_lock(validator, current_time))
            
            time.sleep(30)
            
        except Exception as e:
            bt.logging.error(f"Error in game data update loop: {str(e)}")
            bt.logging.error(traceback.format_exc())
            time.sleep(5)

async def update_game_data_with_lock(validator, current_time):
    """Wrapper to handle the async lock for update_game_data"""
    async with validator.operation_lock:
        # Run the update function in a thread since it's synchronous
        await asyncio.to_thread(update_game_data, validator, current_time)

async def run(validator: BettensorValidator):
    """Main async run loop for the validator"""
    # Load environment variables
    load_dotenv()
    

    
    initialize(validator)
    watchdog = Watchdog(timeout=900)  # 15 minutes timeout

    # Create threads for periodic tasks
    status_log_thread = threading.Thread(target=log_status, args=(validator,), daemon=True)
    game_data_thread = threading.Thread(target=game_data_update_loop, args=(validator,), daemon=True)
    
    # Start the threads
    status_log_thread.start()
    game_data_thread.start()

    # Add state sync tasks
    state_sync_task = asyncio.create_task(push_state_periodic(validator)) #primary node, pushes state to github
    state_check_task = asyncio.create_task(check_state_sync(validator)) #non-primary node, checks if state needs to be pulled from github
    
    try:
        while True:
            current_time = datetime.now(timezone.utc)
            current_block = validator.subtensor.block
            bt.logging.info(f"Current block: {current_block}")

            # Create tasks with timeouts
            tasks = []
            
            # Sync metagraph
            tasks.append(asyncio.create_task(
                run_with_timeout(sync_metagraph_with_retry, METAGRAPH_TIMEOUT, validator)
            ))

            # Check hotkeys
            tasks.append(asyncio.create_task(
                run_with_timeout(validator.check_hotkeys, METAGRAPH_TIMEOUT)
            ))

            # Perform update (if needed)
            tasks.append(asyncio.create_task(
                run_with_timeout(perform_update, UPDATE_TIMEOUT, validator)
            ))

            # Query and process axons
            if (current_block - validator.last_queried_block) > validator.query_axons_interval:
                tasks.append(asyncio.create_task(
                    run_with_timeout(query_and_process_axons_with_game_data, QUERY_TIMEOUT, validator)
                ))

            # Send data to website
            if (current_block - validator.last_sent_data_to_website) > validator.send_data_to_website_interval:
                tasks.append(asyncio.create_task(
                    run_with_timeout(send_data_to_website_server, WEBSITE_TIMEOUT, validator)
                ))

            # Recalculate scores
            if (current_block - validator.last_scoring_block) > validator.scoring_interval:
                tasks.append(asyncio.create_task(
                    run_with_timeout(scoring_run, SCORING_TIMEOUT, validator, current_time)
                ))

            # Set weights
            if (current_block - validator.last_set_weights_block) > validator.set_weights_interval:
                tasks.append(asyncio.create_task(
                    run_with_timeout(set_weights, WEIGHTS_TIMEOUT, validator, validator.scores)
                ))

            # Wait for all tasks to complete
            if tasks:
                completed_tasks = []
                for task in asyncio.as_completed(tasks):
                    try:
                        result = await task
                        completed_tasks.append(result)
                    except asyncio.CancelledError:
                        bt.logging.warning(f"Task was cancelled")
                        continue
                    except Exception as e:
                        bt.logging.error(f"Task failed with error: {str(e)}")
                        continue
                
                # Log completion summary
                bt.logging.info(f"Completed {len(completed_tasks)} out of {len(tasks)} tasks")

            await asyncio.sleep(30)

    except Exception as e:
        bt.logging.error(f"Error in main: {str(e)}")
        bt.logging.error(traceback.format_exc())
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt received. Shutting down gracefully...")
    finally:
        state_sync_task.cancel()
        state_check_task.cancel()
        # Wait for threads to finish
        status_log_thread.join(timeout=1)
        game_data_thread.join(timeout=1)

async def run_with_timeout(func, timeout: int, *args, **kwargs) -> Optional[Any]:
    """
    Enhanced run_with_timeout that ensures proper task cleanup
    """
    try:
        # Convert sync function to async if needed
        if not asyncio.iscoroutinefunction(func):
            async_func = partial(asyncio.to_thread, func)
        else:
            async_func = func
        
        async with async_timeout.timeout(timeout):
            task = asyncio.create_task(async_func(*args, **kwargs))
            try:
                return await task
            except asyncio.CancelledError:
                bt.logging.warning(f"{func.__name__} was cancelled")
                raise
            finally:
                # Ensure task is properly cleaned up
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    
    except asyncio.TimeoutError:
        func_name = getattr(func, '__name__', str(func))
        bt.logging.error(f"{func_name} timed out after {timeout} seconds")
        return None
    except Exception as e:
        func_name = getattr(func, '__name__', str(func))
        bt.logging.error(f"Error in {func_name}: {str(e)}")
        bt.logging.error(traceback.format_exc())
        return None

def initialize(validator):
    validator.is_primary = os.environ.get("VALIDATOR_IS_PRIMARY") == "True"
    should_pull_state = os.environ.get("VALIDATOR_PULL_STATE", "True").lower() == "true"

    # Add state sync initialization
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    ).decode().strip()
    
    #set branch to main if it is none or not main or test
    if branch is None or branch not in ["main", "test"]:
        bt.logging.warning(f"Invalid branch. Setting to main by default for data sync")
        branch = "main"

    validator.state_sync = StateSync(
        state_dir="./bettensor/validator/state",
    )
    
    # Pull latest state before starting if configured to do so
    if should_pull_state:
        bt.logging.info("Pulling latest state from Azure blob storage...")
        if validator.state_sync.pull_state():
            bt.logging.info("Successfully pulled latest state")
        else:
            bt.logging.warning("Failed to pull latest state, continuing with local state")
    else:
        bt.logging.info("Skipping state pull due to VALIDATOR_PULL_STATE configuration")
    
    validator.serve_axon()
    validator.initialize_connection()

    if not validator.last_updated_block:
        bt.logging.info("Updating last updated block; will set weights this iteration")
        validator.last_updated_block = validator.subtensor.block - 301
        validator.last_queried_block = validator.subtensor.block - 11
        validator.last_sent_data_to_website = validator.subtensor.block - 16
        validator.last_scoring_block = validator.subtensor.block - 51
        validator.last_set_weights_block = validator.subtensor.block - 301
    
    # Define default intervals if they don't exist
    if not hasattr(validator, 'update_game_data_interval'):
        validator.update_game_data_interval = 10  # Default value, adjust as needed

    if not hasattr(validator, 'query_axons_interval'):
        validator.query_axons_interval = 40 # Default value, adjust as needed

    if not hasattr(validator, 'send_data_to_website_interval'):
        validator.send_data_to_website_interval = 15  # Default value, adjust as needed

    if not hasattr(validator, 'scoring_interval'):
        validator.scoring_interval = 60  # Default value, adjust as needed

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
    validator.operation_lock = asyncio.Lock()

# Add periodic state pushing for primary node
async def push_state_periodic(validator):
    """Periodically push state files to Azure blob storage if primary node"""
    bt.logging.info("--------------------------------Pushing state files to Azure blob storage--------------------------------")
    while True:
        if validator.is_primary:
            bt.logging.debug("Primary node, pushing state files to Azure blob storage")
            if validator.state_sync.push_state():
                bt.logging.info("Successfully pushed state files")
            else:
                bt.logging.error("Failed to push state files")
        else:
            bt.logging.info("Not primary node, skipping state push")
        await asyncio.sleep(3600)  # Push every hour

@time_task("update_game_data")
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

@time_task("sync_metagraph")
async def sync_metagraph_with_retry(validator):
    max_retries = 3
    retry_delay = 60
    for attempt in range(max_retries):
        try:
            validator.metagraph = validator.sync_metagraph()
            bt.logging.info("Metagraph synced successfully.")
            return
        except websocket.WebSocketConnectionClosedException:
            if attempt < max_retries - 1:
                bt.logging.warning(f"WebSocket connection closed. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                raise
        except Exception as e:
            bt.logging.error(f"Error syncing metagraph: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise

@time_task("filter_and_update_axons")
async def filter_and_update_axons(validator):
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

    # Run get_uids_to_query in a thread since it might be CPU-bound
    result = await asyncio.to_thread(
        validator.get_uids_to_query,
        all_axons=all_axons
    )
    
    # Make sure we're returning a tuple of values, not a coroutine
    return result

@time_task("query_and_process_axons")
async def query_and_process_axons_with_game_data(validator):
    """Queries axons with game data and processes the responses"""
    bt.logging.info("--------------------------------Querying and processing axons with game data--------------------------------")
    
    async with validator.operation_lock:  # Add lock here
        validator.last_queried_block = validator.subtensor.block
        current_time = datetime.now(timezone.utc).isoformat()
        
        gamedata_dict = await asyncio.to_thread(
            validator.fetch_local_game_data,
            current_timestamp=current_time
        )
        
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

        result = await filter_and_update_axons(validator)
        if result is None:
            bt.logging.error("Failed to filter and update axons")
            return None

        uids_to_query, list_of_uids, blacklisted_uids, uids_not_to_query = result
        
        # Define batch size
        BATCH_SIZE = 10  # Adjust based on testing
        responses = []
        
        try:
            # Process axons in batches
            for i in range(0, len(uids_to_query), BATCH_SIZE):
                batch = uids_to_query[i:i + BATCH_SIZE]
                batch_uids = list_of_uids[i:i + BATCH_SIZE]
                bt.logging.debug(f"Processing batch {i//BATCH_SIZE + 1} of {math.ceil(len(uids_to_query)/BATCH_SIZE)}")
                
                # Query batch of axons
                batch_responses = await asyncio.to_thread(
                    validator.dendrite.query,
                    axons=batch,
                    synapse=synapse,
                    timeout=validator.timeout,
                    deserialize=False,
                )
                
                # Process valid responses from this batch
                valid_batch_responses = []
                if batch_responses and hasattr(batch_responses, '__iter__'):
                    for response in batch_responses:
                        if isinstance(response, GameData):
                            if response.metadata.synapse_type == "prediction":
                                valid_batch_responses.append(response)
                
                # Process predictions for this batch immediately
                bt.logging.debug(f"Processing {len(valid_batch_responses)} predictions for batch {i//BATCH_SIZE + 1}")
                if valid_batch_responses:
                    try:
                        validator.process_prediction(processed_uids=batch_uids, synapses=valid_batch_responses)
                        responses.extend(valid_batch_responses)
                    except Exception as e:
                        bt.logging.error(f"Error processing batch predictions: {str(e)}")
                        bt.logging.error(traceback.format_exc())
                
                # Optional: Add a small delay between batches
                await asyncio.sleep(0.1)
                
            
            return len(responses)

        except Exception as e:
            bt.logging.error(f"Error processing responses: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return None

@time_task("send_data_to_website_server")
async def send_data_to_website_server(validator):
    """
    Sends data to the website server
    """
    bt.logging.info("--------------------------------Sending data to website server--------------------------------")
    validator.last_sent_data_to_website = validator.subtensor.block
    bt.logging.info(f"Last sent data to website: {validator.last_sent_data_to_website}")

    try:
        result = validator.website_handler.fetch_and_send_predictions()
        bt.logging.info(f"Result status: {result}")
        if result:
            bt.logging.info("Predictions fetched and sent successfully")
        else:
            bt.logging.warning("No new predictions were sent this round")
    except Exception as e:
        bt.logging.error(f"Error in fetch_and_send_predictions: {str(e)}")

@time_task("scoring_run")
async def scoring_run(validator, current_time):
    """
    calls the scoring system to update miner scores before setting weights
    """
    bt.logging.info("--------------------------------Scoring run--------------------------------")
    validator.last_scoring_block = validator.subtensor.block
    
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
        bt.logging.info(f"Invalid UIDs: {invalid_uids}")
        validator.scores = validator.scoring_system.scoring_run(
            current_time, invalid_uids, valid_uids
        )
        bt.logging.info("Scores updated successfully")
        bt.logging.info(f"Scores: {validator.scores}")

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
        bt.logging.info(f"Scoring run completed")

    except Exception as e:
        bt.logging.error(f"Error in scoring_run: {str(e)}")
        bt.logging.error(f"Traceback: {traceback.format_exc()}")
        raise



@time_task("set_weights")
async def set_weights(validator, weights_to_set):
    """Wrapper for weight setting that handles the multiprocessing timeout gracefully"""
    try:
        # Run the weight setter in a thread to not block the event loop
        result = await asyncio.to_thread(
            validator.weight_setter.set_weights,
            weights_to_set
        )
        return result
    except Exception as e:
        bt.logging.error(f"Error in set_weights wrapper: {str(e)}")
        bt.logging.error(traceback.format_exc())
        return False


@time_task("check_state_sync")
async def check_state_sync(validator):
    """Periodically check and sync state if needed"""
    while True:
        try:
            if not validator.is_primary:
                if validator.state_sync.should_pull_state():
                    bt.logging.info("State divergence detected, pulling latest state")
                    if validator.state_sync.pull_state():
                        bt.logging.info("Successfully pulled latest state")
                    else:
                        bt.logging.error("Failed to pull latest state")
            await asyncio.sleep(3600)  # Check every hour
        except Exception as e:
            bt.logging.error(f"Error in state sync check: {e}")
            await asyncio.sleep(300)  # On error, retry after 5 minutes

def cleanup_pycache():
    """Remove all __pycache__ directories and .pyc files"""
    bt.logging.info("Cleaning up __pycache__ directories and .pyc files")
    try:
        # Get the root directory (where the script is running)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Walk through all directories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Remove __pycache__ directories
            if '__pycache__' in dirnames:
                cache_path = os.path.join(dirpath, '__pycache__')
                try:
                    bt.logging.debug(f"Removing {cache_path}")
                    import shutil
                    shutil.rmtree(cache_path)
                except Exception as e:
                    bt.logging.warning(f"Failed to remove {cache_path}: {str(e)}")
            
            # Remove .pyc files
            for filename in filenames:
                if filename.endswith('.pyc'):
                    pyc_path = os.path.join(dirpath, filename)
                    try:
                        bt.logging.debug(f"Removing {pyc_path}")
                        os.remove(pyc_path)
                    except Exception as e:
                        bt.logging.warning(f"Failed to remove {pyc_path}: {str(e)}")
                        
    except Exception as e:
        bt.logging.error(f"Error during pycache cleanup: {str(e)}")

# The main function parses the configuration and runs the validator.
def main():
    # Add cleanup at the start of main
    cleanup_pycache()
    
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
        help="Sets the value for the number of targets to query - set to 256 to ensure all miners are queried, it is now batched",
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

    # Run the async main loop
    asyncio.run(run(validator))





if __name__ == "__main__":
    ensure_dependencies()
    main()