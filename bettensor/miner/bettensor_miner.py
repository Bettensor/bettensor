import json
import signal
import sys
from argparse import ArgumentParser
import time
import traceback
from typing import Tuple
import bittensor as bt
import sqlite3
from bettensor.base.neuron import BaseNeuron
from bettensor.protocol import Metadata, GameData, TeamGame, TeamGamePrediction
from bettensor.miner.stats.miner_stats import MinerStateManager, MinerStatsHandler
import datetime
import os
import threading
from contextlib import contextmanager
from bettensor.miner.database.database_manager import DatabaseManager
from bettensor.miner.database.games import GamesHandler
from bettensor.miner.database.predictions import PredictionsHandler
from bettensor.miner.utils.cache_manager import CacheManager
from bettensor.miner.interfaces.redis_interface import RedisInterface
from bettensor.miner.models.model_utils import SoccerPredictor, MinerConfig
import uuid
from datetime import datetime, timezone

class BettensorMiner(BaseNeuron):
    def __init__(self, parser: ArgumentParser):
        bt.logging.info("Initializing BettensorMiner")
        super().__init__(parser=parser, profile="miner")
        
        bt.logging.info("Adding custom arguments")
        
        # Add PostgreSQL connection parameters with defaults
        parser.add_argument("--db_name", type=str, default="bettensor", help="PostgreSQL database name")
        parser.add_argument("--db_user", type=str, default="root", help="PostgreSQL user")
        parser.add_argument("--db_password", type=str, default="bettensor_password", help="PostgreSQL password")
        parser.add_argument("--db_host", type=str, default="localhost", help="PostgreSQL host")
        parser.add_argument("--db_port", type=int, default=5432, help="PostgreSQL port")
        parser.add_argument("--max_connections", type=int, default=10, help="Maximum number of database connections")
        
        if not any(action.dest == 'validator_min_stake' for action in parser._actions):
            parser.add_argument("--validator_min_stake", type=float, default=1000.0, help="Minimum stake required for validators")
        
        if not any(action.dest == 'redis_host' for action in parser._actions):
            parser.add_argument("--redis_host", type=str, default="localhost", help="Redis server host")
        
        if not any(action.dest == 'redis_port' for action in parser._actions):
            parser.add_argument("--redis_port", type=int, default=6379, help="Redis server port")
        
        bt.logging.info("Parsing arguments and setting up configuration")
        try:
            self.neuron_config = self.config(bt_classes=[bt.subtensor, bt.logging, bt.wallet, bt.axon])
            if self.neuron_config is None:
                raise ValueError("self.config() returned None")
        except Exception as e:
            bt.logging.error(f"Error in self.config(): {e}")
            raise

        bt.logging.info(f"Neuron config: {self.neuron_config}")

        self.args = self.neuron_config

        bt.logging.info("Setting up wallet, subtensor, and metagraph")
        try:
            self.wallet, self.subtensor, self.metagraph = self.setup()
            self.miner_uid = str(self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address))
            bt.logging.info(f"Miner initialized with UID: {self.miner_uid}")
        except Exception as e:
            bt.logging.error(f"Error in self.setup(): {e}")
            raise

        # Initialize Redis interface
        self.redis_interface = RedisInterface(host=self.args.redis_host, port=self.args.redis_port)
        if not self.redis_interface.connect():
            bt.logging.warning("Failed to connect to Redis. GUI interfaces will not be available.")
            self.gui_available = False
        else:
            bt.logging.info("Redis connection successful. All interfaces (GUI and CLI) are available.")
            self.gui_available = True
            # Start Redis listener in a separate thread
            self.redis_thread = threading.Thread(target=self.listen_for_redis_messages)
            self.redis_thread.daemon = True
            self.redis_thread.start()

        # Setup database manager
        bt.logging.info("Initializing database manager")
        db_params = {
            "db_name": self.args.db_name,
            "db_user": self.args.db_user,
            "db_password": self.args.db_password,
            "db_host": self.args.db_host,
            "db_port": self.args.db_port,
            "max_connections": self.args.max_connections
        }
        self.db_manager = DatabaseManager(**db_params)
        self.db_manager.initialize_default_model_params(miner_uid=self.miner_uid)
        # Setup state manager
        bt.logging.info("Initializing state manager")
        self.miner_hotkey = self.wallet.hotkey.ss58_address
        self.miner_uid = self.miner_uid

        self.state_manager = MinerStateManager(
            db_manager=self.db_manager,
            miner_hotkey=self.miner_hotkey,
            miner_uid=self.miner_uid
        )

        # Create an instance of MinerStatsHandler
        self.stats_handler = MinerStatsHandler(self.state_manager)

        # Setup handlers
        bt.logging.info("Initializing handlers")
        self.predictions_handler = PredictionsHandler(self.db_manager, self.state_manager, self.miner_hotkey)
        self.games_handler = GamesHandler(self.db_manager, self.predictions_handler)

        # Setup cache manager
        bt.logging.info("Initializing cache manager")
        self.cache_manager = CacheManager()

        # Setup other attributes
        bt.logging.info("Setting other attributes")
        self.validator_min_stake = self.args.validator_min_stake
        self.hotkey = self.wallet.hotkey.ss58_address

        bt.logging.info("Setting up signal handlers")
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.hotkey_blacklisted = False

        bt.logging.info("BettensorMiner initialization complete")

        self.last_incentive_update = None
        self.incentive_update_interval = 600  # Update every 10 minutes

        # Initialize MinerConfig
        self.miner_config = MinerConfig()

        # Initialize SoccerPredictor
        #self.soccer_predictor = SoccerPredictor(model_name="your_model_name_here")

    def forward(self, synapse: GameData) -> GameData:
        bt.logging.info(f"Miner: Received synapse from {synapse.dendrite.hotkey}")

        # Print version information and perform version checks
        print(f"Synapse version: {synapse.metadata.subnet_version}, our version: {self.subnet_version}")
        self._check_version(synapse.metadata.subnet_version)

        bt.logging.debug(f"Processing game data: {len(synapse.gamedata_dict)} games")
        for game in synapse.gamedata_dict.values():
            bt.logging.debug(f"Game: {game}")
        try:
            # Check if there are any changes in game data
            changed_games = self.cache_manager.filter_changed_games(synapse.gamedata_dict)

            if changed_games:
                bt.logging.info(f"Processing {len(changed_games)} changed games")
                updated_games, new_games = self.games_handler.process_games(changed_games)
                recent_predictions = self.predictions_handler.process_predictions(updated_games, new_games)
                
                # Update cache with new predictions
                self.cache_manager.update_cached_predictions(recent_predictions)
            else:
                bt.logging.info("No changes in game data, using cached predictions")
                recent_predictions = self.cache_manager.get_cached_predictions()

            if not recent_predictions:
                bt.logging.warning("No predictions available")
                return self._clean_synapse(synapse)


            synapse.prediction_dict = recent_predictions
            synapse.gamedata_dict = None
            synapse.metadata = self._create_metadata("prediction")

            bt.logging.info(f"Number of predictions added to synapse: {len(recent_predictions)}")
            
        except Exception as e:
            bt.logging.error(f"Error in forward method: {e}")
            return self._clean_synapse(synapse)

        return synapse

    def _check_version(self, synapse_version):
        if synapse_version > self.subnet_version:
            bt.logging.warning(
                f"Received a synapse from a validator with higher subnet version ({synapse_version}) than yours ({self.subnet_version}). Please update the miner, or you may encounter issues."
            )
        elif synapse_version < self.subnet_version:
            bt.logging.warning(
                f"Received a synapse from a validator with lower subnet version ({synapse_version}) than yours ({self.subnet_version}). You can safely ignore this warning."
            )

    def _create_metadata(self, synapse_type):
        return Metadata.create(
            wallet=self.wallet,
            subnet_version=self.subnet_version,
            neuron_uid=self.miner_uid,
            synapse_type=synapse_type,
        )

    def _clean_synapse(self, synapse: GameData) -> GameData:
        if not synapse.prediction_dict:
            bt.logging.debug("Cleaning synapse due to no predictions available")
        else:
            bt.logging.debug("Cleaning synapse due to error")
        
        synapse.gamedata_dict = None
        synapse.prediction_dict = None
        synapse.metadata = Metadata.create(
            wallet=self.wallet,
            subnet_version=self.subnet_version,
            neuron_uid=self.miner_uid,
            synapse_type="error",
        )
        bt.logging.debug("Synapse cleaned")
        return synapse

    def start(self):
        bt.logging.info("Starting miner")
        self.stats_handler.reset_daily_cash()
        
        # Start Redis listener in a separate thread
        self.redis_thread = threading.Thread(target=self.listen_for_redis_messages)
        self.redis_thread.daemon = True
        self.redis_thread.start()
        
        # Start health check in a separate thread
        self.health_thread = threading.Thread(target=self.health_check)
        self.health_thread.daemon = True
        self.health_thread.start()
        
        bt.logging.info("Miner started")

    def stop(self):
        bt.logging.info("Stopping miner")
        try:
            current_state = self.state_manager.get_stats()
            self.state_manager.save_state(current_state)
        except Exception as e:
            bt.logging.error(f"Error saving state: {e}")
        bt.logging.info("Miner stopped")

    def signal_handler(self, signum, frame):
        bt.logging.info(f"Received signal {signum}. Shutting down...")
        self.stop()
        bt.logging.info("Exiting due to signal")
        sys.exit(0)

    def setup(self) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph]:
        bt.logging.info("Setting up bittensor objects")
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        bt.logging.info(
            f"Initializing miner for subnet: {self.neuron_config.netuid} on network: {self.neuron_config.subtensor.chain_endpoint} with config:\n {self.neuron_config}"
        )

        try:
            wallet = bt.wallet(config=self.neuron_config)
            subtensor = bt.subtensor(config=self.neuron_config)
            metagraph = subtensor.metagraph(self.neuron_config.netuid)
        except AttributeError as e:
            bt.logging.error(f"Unable to setup bittensor objects: {e}")
            sys.exit()

        bt.logging.info(
            f"Bittensor objects initialized:\nMetagraph: {metagraph}\
            \nSubtensor: {subtensor}\nWallet: {wallet}"
        )

        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"Your miner: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again"
            )
            sys.exit()

        bt.logging.info("Bittensor objects setup complete")
        return wallet, subtensor, metagraph

    def check_whitelist(self, hotkey):
        bt.logging.debug(f"Checking whitelist for hotkey: {hotkey}")
        if isinstance(hotkey, bool) or not isinstance(hotkey, str):
            bt.logging.debug(f"Invalid hotkey type: {type(hotkey)}")
            return False

        whitelisted_hotkeys = []

        if hotkey in whitelisted_hotkeys:
            bt.logging.debug(f"Hotkey {hotkey} is whitelisted")
            return True

        bt.logging.debug(f"Hotkey {hotkey} is not whitelisted")
        return False

    def blacklist(self, synapse: GameData) -> Tuple[bool, str]:
        bt.logging.debug(f"Checking blacklist for synapse from {synapse.dendrite.hotkey}")
        if self.check_whitelist(hotkey=synapse.dendrite.hotkey):
            bt.logging.info(f"Accepted whitelisted hotkey: {synapse.dendrite.hotkey}")
            return (False, f"Accepted whitelisted hotkey: {synapse.dendrite.hotkey}")

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.info(f"Blacklisted unknown hotkey: {synapse.dendrite.hotkey}")
            return (
                True,
                f"Hotkey {synapse.dendrite.hotkey} was not found from metagraph.hotkeys",
            )

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.metagraph.validator_permit[uid]:
            bt.logging.info(f"Blacklisted non-validator: {synapse.dendrite.hotkey}")
            return (True, f"Hotkey {synapse.dendrite.hotkey} is not a validator")

        bt.logging.info(f"validator_min_stake: {self.validator_min_stake}")
        stake = float(self.metagraph.S[uid])
        if stake < self.validator_min_stake:
            bt.logging.info(
                f"Blacklisted validator {synapse.dendrite.hotkey} with insufficient stake: {stake}"
            )
            return (
                True,
                f"Hotkey {synapse.dendrite.hotkey} has insufficient stake: {stake}",
            )

        bt.logging.info(
            f"Accepted hotkey: {synapse.dendrite.hotkey} (UID: {uid} - Stake: {stake})"
        )
        return (False, f"Accepted hotkey: {synapse.dendrite.hotkey}")

    def priority(self, synapse: GameData) -> float:
        bt.logging.debug(f"Calculating priority for synapse from {synapse.dendrite.hotkey}")
        if self.check_whitelist(hotkey=synapse.dendrite.hotkey):
            bt.logging.debug(f"Whitelisted hotkey {synapse.dendrite.hotkey}, returning max priority")
            return 10000000.0

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = float(self.metagraph.S[uid])

        bt.logging.debug(f"Prioritized: {synapse.dendrite.hotkey} (UID: {uid} - Stake: {stake})")
        return stake

    def get_current_incentive(self):
        current_time = time.time()
        
        # Check if it's time to update the incentive
        if self.last_incentive_update is None or (current_time - self.last_incentive_update) >= self.incentive_update_interval:
            bt.logging.info("Updating current incentive")
            try:
                # Sync the metagraph to get the latest data
                self.metagraph.sync()
                
                # Get the incentive for this miner
                miner_uid_int = int(self.miner_uid)
                incentive = self.metagraph.I[miner_uid_int].item() if miner_uid_int < len(self.metagraph.I) else 0.0
                
                # Update the stats handler with the new incentive
                self.stats_handler.update_current_incentive(incentive)
                
                self.last_incentive_update = current_time
                
                bt.logging.info(f"Updated current incentive to: {incentive}")
                return incentive
            except Exception as e:
                bt.logging.error(f"Error updating current incentive: {e}")
                return None
        else:
            # If it's not time to update, return the last known incentive from the stats handler
            return self.stats_handler.get_current_incentive()

    def listen_for_redis_messages(self):
        channel = f'miner:{self.miner_uid}:{self.wallet.hotkey.ss58_address}'
        bt.logging.info(f"Starting to listen for Redis messages on channel: {channel}")
        
        while True:
            try:
                pubsub = self.redis_interface.subscribe(channel)
                if pubsub is None:
                    bt.logging.error("Failed to subscribe to Redis channel")
                    time.sleep(5)  # Wait before trying to reconnect
                    continue

                bt.logging.info(f"Successfully subscribed to Redis channel: {channel}")

                for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            bt.logging.info(f"Received message: {data}")
                            
                            action = data.get('action')
                            if action is None:
                                bt.logging.warning(f"Received message without 'action' field: {data}")
                                continue
                            
                            if action == 'make_prediction':
                                result = self.process_prediction_request(data)
                                
                                # Send the result back
                                response_key = f'response:{data.get("message_id", "unknown")}'
                                bt.logging.info(f"Publishing response to key: {response_key}")
                                self.redis_interface.set(response_key, json.dumps(result), ex=60)  # Set expiration to 60 seconds
                            else:
                                bt.logging.warning(f"Unknown action: {action}")
                        except json.JSONDecodeError:
                            bt.logging.error(f"Failed to decode JSON message: {message['data']}")
                        except KeyError as e:
                            bt.logging.error(f"Missing key in message: {e}")
                        except Exception as e:
                            bt.logging.error(f"Error processing message: {str(e)}")

            except Exception as e:
                bt.logging.error(f"Error in Redis listener: {str(e)}")
                time.sleep(5)  # Wait before trying to reconnect

    def process_prediction_request(self, data):
        bt.logging.info(f"Processing prediction request: {data}")

        predictions = data.get('predictions')
        if not predictions:
            bt.logging.warning("Received prediction request without 'predictions' field")
            return {
                'status': 'error',
                'message': "No predictions provided",
                'miner_uid': self.miner_uid,
                'miner_hotkey': self.wallet.hotkey.ss58_address
            }

        results = []

        for prediction in predictions:
            bt.logging.info(f"Processing prediction: {prediction}")
            
            try:
                # Check if the game exists using the externalID
                game_exists = self.games_handler.game_exists(prediction.get('externalID'))
                if not game_exists:
                    bt.logging.warning(f"Game with ID {prediction.get('externalID')} does not exist")
                    results.append({
                        'status': 'error',
                        'message': f"Game with ID {prediction.get('externalID')} does not exist"
                    })
                    continue

                # Generate a new predictionID
                prediction['predictionID'] = str(uuid.uuid4())

                # Set minerID and ensure predictionDate is in the correct format
                prediction['minerID'] = self.miner_uid
                prediction['predictionDate'] = datetime.fromisoformat(prediction['predictionDate']).replace(tzinfo=timezone.utc).isoformat()

                # Set initial outcome to 'Unfinished'
                prediction['outcome'] = 'Unfinished'

                # Add the prediction to the database
                result = self.predictions_handler.add_prediction(prediction)
                bt.logging.info(f"Prediction added: {result}")
                results.append(result)

            except Exception as e:
                bt.logging.error(f"Error processing prediction: {str(e)}")
                bt.logging.error(f"Traceback: {traceback.format_exc()}")
                results.append({
                    'status': 'error',
                    'message': f"Error processing prediction: {str(e)}"
                })

        return {
            'status': 'success',
            'message': "Predictions processed",
            'results': results,
            'miner_uid': self.miner_uid,
            'miner_hotkey': self.wallet.hotkey.ss58_address
        }

    def health_check(self):
        while True:
            bt.logging.info("Miner health check: Still listening for Redis messages")
            time.sleep(300)  # Check every 5 minutes