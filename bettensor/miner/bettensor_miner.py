import signal
import sys
from argparse import ArgumentParser
import time
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
from bettensor.miner.database.database_manager import get_db_manager
from bettensor.miner.database.games import GamesHandler
from bettensor.miner.database.predictions import PredictionsHandler

class BettensorMiner(BaseNeuron):
    def __init__(self, parser: ArgumentParser):
        bt.logging.info("Initializing BettensorMiner")
        super().__init__(parser=parser, profile="miner")
        
        bt.logging.info("Adding custom arguments")
        self.default_db_path = os.path.expanduser("~/bettensor/data/miner.db")
        
        if not any(action.dest == 'db_path' for action in parser._actions):
            parser.add_argument("--db_path", type=str, default=self.default_db_path, help="Path to the SQLite database file")
        
        if not any(action.dest == 'max_connections' for action in parser._actions):
            parser.add_argument("--max_connections", type=int, default=10, help="Maximum number of database connections")
        
        if not any(action.dest == 'validator_min_stake' for action in parser._actions):
            parser.add_argument("--validator_min_stake", type=float, default=1000.0, help="Minimum stake required for validators")
        
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
            self.wallet, self.subtensor, self.metagraph, self.miner_uid = self.setup()
        except Exception as e:
            bt.logging.error(f"Error in self.setup(): {e}")
            raise

        bt.logging.info("Initializing database manager")
        os.environ['DB_PATH'] = self.args.db_path
        bt.logging.info(f"Set DB_PATH environment variable to: {self.args.db_path}")
        try:
            bt.logging.info(f"Calling get_db_manager with max_connections: {self.args.max_connections}")
            self.db_manager = get_db_manager(
                max_connections=self.args.max_connections,
                state_manager=None,
                miner_uid=self.miner_uid
            )
            bt.logging.info("Database manager initialized successfully")
        except Exception as e:
            bt.logging.error(f"Failed to initialize database manager: {e}")
            raise

        bt.logging.info("Initializing state manager")
        self.state_manager = MinerStateManager(
            db_manager=self.db_manager,
            miner_hotkey=self.wallet.hotkey.ss58_address,
            miner_uid=self.miner_uid
        )
        
        bt.logging.info("Initializing handlers")
        self.games_handler = GamesHandler(self.db_manager)
        self.predictions_handler = PredictionsHandler(self.db_manager, self.state_manager, self.miner_uid)
        
        bt.logging.info("Setting other attributes")
        self.validator_min_stake = self.args.validator_min_stake
        self.hotkey = self.wallet.hotkey.ss58_address
        
        bt.logging.info("Setting up signal handlers")
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        bt.logging.info(f"Miner initialized with UID: {self.miner_uid}")

        self.hotkey_blacklisted = False
        
        bt.logging.info("BettensorMiner initialization complete")

        self.last_incentive_update = None
        self.incentive_update_interval = 600  # Update every 10 minutes

    def forward(self, synapse: GameData) -> GameData:
        bt.logging.info(f"Miner: Received synapse from {synapse.dendrite.hotkey}")

        # Print version information and perform version checks
        print(
            f"Synapse version: {synapse.metadata.subnet_version}, our version: {self.subnet_version}"
        )
        if synapse.metadata.subnet_version > self.subnet_version:
            bt.logging.warning(
                f"Received a synapse from a validator with higher subnet version ({synapse.metadata.subnet_version}) than yours ({self.subnet_version}). Please update the miner, or you may encounter issues."
            )
        if synapse.metadata.subnet_version < self.subnet_version:
            bt.logging.warning(
                f"Received a synapse from a validator with lower subnet version ({synapse.metadata.subnet_version}) than yours ({self.subnet_version}). You can safely ignore this warning."
            )


        bt.logging.debug(f"Processing game data: {len(synapse.gamedata_dict)} games")

        try:
            # Process games and create new predictions
            updated_games, new_games = self.games_handler.process_games(synapse.gamedata_dict)
            recent_predictions = self.predictions_handler.process_predictions(updated_games, new_games)

            if not recent_predictions:
                bt.logging.warning("Failed to process games")
                return self._clean_synapse(synapse)

            # Update miner stats
            self.state_manager.update_stats_from_predictions(recent_predictions.values(), updated_games)

            # Periodic database update
            self.state_manager.periodic_db_update()

            synapse.prediction_dict = recent_predictions
            synapse.gamedata_dict = None
            synapse.metadata = Metadata.create(
                wallet=self.wallet,
                subnet_version=self.subnet_version,
                neuron_uid=self.miner_uid,
                synapse_type="prediction",
            )

            return synapse
        except Exception as e:
            bt.logging.error(f"Error in forward method: {e}")
            return self._clean_synapse(synapse)

    def _clean_synapse(self, synapse: GameData) -> GameData:
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
        self.state_manager.reset_daily_cash()
        bt.logging.info("Miner started")

    def stop(self):
        bt.logging.info("Stopping miner")
        self.state_manager.save_state()
        bt.logging.info("Miner stopped")

    def signal_handler(self, signum, frame):
        bt.logging.info(f"Received signal {signum}. Shutting down...")
        self.stop()
        bt.logging.info("Exiting due to signal")
        sys.exit(0)

    def setup(self) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph, str]:
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

        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Miner is running with UID: {miner_uid}")

        bt.logging.info("Bittensor objects setup complete")
        return wallet, subtensor, metagraph, miner_uid

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
                incentive = float(self.metagraph.I[self.miner_uid])
                
                # Update the state manager with the new incentive
                self.state_manager.update_current_incentive(incentive)
                
                self.last_incentive_update = current_time
                
                bt.logging.info(f"Updated current incentive to: {incentive}")
                return incentive
            except Exception as e:
                bt.logging.error(f"Error updating current incentive: {e}")
                return None
        else:
            # If it's not time to update, return the last known incentive from the state manager
            return self.state_manager.get_current_incentive()