import signal
import sys
from argparse import ArgumentParser
from typing import Tuple
import bittensor as bt
import sqlite3
from bettensor.base.neuron import BaseNeuron
from bettensor.protocol import Metadata, GameData, TeamGame, TeamGamePrediction
from bettensor.utils.sign_and_validate import verify_signature
from bettensor.utils.miner_stats import MinerStateManager
import datetime
import os
import threading
from contextlib import contextmanager
from bettensor.miner.database.database_manager import get_db_manager
from bettensor.miner.database.games import GamesHandler
from bettensor.miner.database.predictions import PredictionsHandler

class BettensorMiner(BaseNeuron):
    def __init__(self, parser: ArgumentParser):
        """
        Initialize the BettensorMiner.

        Args:
            parser (ArgumentParser): The argument parser for the miner.

        Behavior:
            - Sets up the miner configuration
            - Initializes wallet, subtensor, and metagraph
            - Sets up database manager and state manager
            - Initializes game and prediction handlers
            - Sets up signal handlers for graceful shutdown
        """
        bt.logging.trace("Initializing BettensorMiner")
        super().__init__(parser=parser, profile="miner")
        
        bt.logging.trace("Adding custom arguments")
        parser.add_argument("--db_path", type=str, default=self.default_db_path)
        parser.add_argument("--validator_min_stake", type=float, default=1000.0)
        
        bt.logging.trace("Parsing arguments and setting up configuration")
        self.neuron_config = self.config(bt_classes=[bt.subtensor, bt.logging, bt.wallet, bt.axon])
        self.args = self.neuron_config.parse_args()
        
        bt.logging.trace("Setting up wallet, subtensor, and metagraph")
        self.wallet, self.subtensor, self.metagraph, self.miner_uid = self.setup()
        
        bt.logging.trace("Initializing database manager")
        self.db_manager = get_db_manager(max_connections=10)  # Adjust max_connections as needed
        
        bt.logging.trace("Initializing state manager")
        self.state_manager = MinerStateManager(
            db_manager=self.db_manager,
            hotkey=self.wallet.hotkey.ss58_address,
            uid=self.miner_uid
        )
        
        bt.logging.trace("Initializing miner stats")
        self.state_manager.stats_handler.init_miner_row()
        self.state_manager.stats_handler.reset_daily_cash_on_startup()
        
        bt.logging.trace("Initializing handlers")
        self.games_handler = GamesHandler(self.db_manager)
        self.predictions_handler = PredictionsHandler(self.db_manager, self.state_manager, self.miner_uid)
        
        bt.logging.trace("Setting other attributes")
        self.validator_min_stake = self.args.validator_min_stake
        self.hotkey = self.wallet.hotkey.ss58_address
        
        bt.logging.trace("Setting up signal handlers")
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        bt.logging.info(f"Miner initialized with UID: {self.miner_uid}")

    def forward(self, synapse: GameData) -> GameData:
        """
        Process an incoming synapse and generate a response.

        Args:
            synapse (GameData): The incoming synapse containing game data.

        Returns:
            GameData: The response synapse containing prediction data.

        Behavior:
            - Processes incoming game data
            - Generates predictions for games
            - Updates the miner's state
            - Prepares and returns a response synapse
        """
        bt.logging.trace(f"Forward method called with synapse from {synapse.dendrite.hotkey}")

        if synapse.metadata.subnet_version > self.subnet_version:
            bt.logging.warning(f"Received a synapse from a validator with higher subnet version ({synapse.metadata.subnet_version}) than yours ({self.subnet_version}). Please update the miner.")
        
        game_data_dict = synapse.gamedata_dict
        bt.logging.info(f"Processing game data: {len(game_data_dict)} games")

        bt.logging.trace("Processing games")
        updated_games, new_games = self.games_handler.process_games(game_data_dict)

        bt.logging.trace("Processing predictions")
        new_prediction_dict = self.predictions_handler.process_predictions(updated_games, new_games)

        if new_prediction_dict is None:
            bt.logging.error("Failed to process games")
            bt.logging.trace("Cleaning up synapse due to processing failure")
            return self._clean_synapse(synapse)

        bt.logging.trace("Updating database with current state")
        self.state_manager.periodic_db_update()

        bt.logging.trace("Preparing synapse for return")
        synapse.prediction_dict = new_prediction_dict
        synapse.gamedata_dict = None
        synapse.metadata = Metadata.create(
            wallet=self.wallet,
            subnet_version=self.subnet_version,
            neuron_uid=self.miner_uid,
            synapse_type="prediction",
        )

        bt.logging.trace("Forward method complete")
        return synapse

    def _clean_synapse(self, synapse: GameData) -> GameData:
        """
        Clean a synapse in case of an error.

        Args:
            synapse (GameData): The synapse to clean.

        Returns:
            GameData: The cleaned synapse.

        Behavior:
            - Removes sensitive data from the synapse
            - Updates the synapse metadata to indicate an error
        """
        bt.logging.trace("Cleaning synapse due to error")
        synapse.gamedata_dict = None
        synapse.prediction_dict = None
        synapse.metadata = Metadata.create(
            wallet=self.wallet,
            subnet_version=self.subnet_version,
            neuron_uid=self.miner_uid,
            synapse_type="error",
        )
        bt.logging.trace("Synapse cleaned")
        return synapse

    def start(self):
        """
        Start the miner.

        Behavior:
            - Resets the daily cash
            - Performs any necessary startup procedures
        """
        bt.logging.info("Starting miner")
        self.state_manager.reset_daily_cash()
        bt.logging.trace("Miner started")

    def stop(self):
        """
        Stop the miner.

        Behavior:
            - Saves the current state
            - Performs any necessary cleanup
        """
        bt.logging.info("Stopping miner")
        self.state_manager.save_state()
        bt.logging.trace("Miner stopped")

    def signal_handler(self, signum, frame):
        """
        Handle shutdown signals.

        Args:
            signum: The signal number.
            frame: The current stack frame.

        Behavior:
            - Logs the received signal
            - Calls the stop method
            - Exits the program
        """
        bt.logging.info(f"Received signal {signum}. Shutting down...")
        self.stop()
        bt.logging.trace("Exiting due to signal")
        sys.exit(0)

    def setup(self) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph, str]:
        """
        Set up the bittensor objects needed for the miner.

        Returns:
            Tuple[bt.wallet, bt.subtensor, bt.metagraph, str]: A tuple containing the wallet, subtensor, metagraph, and miner UID.

        Behavior:
            - Initializes wallet, subtensor, and metagraph
            - Verifies that the miner is registered on the network
            - Determines the miner's UID
        """
        bt.logging.trace("Setting up bittensor objects")
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

        bt.logging.trace("Bittensor objects setup complete")
        return wallet, subtensor, metagraph, miner_uid

    def check_whitelist(self, hotkey):
        """
        Check if a hotkey is whitelisted.

        Args:
            hotkey: The hotkey to check.

        Returns:
            bool: True if the hotkey is whitelisted, False otherwise.

        Behavior:
            - Verifies the hotkey type
            - Checks against a list of whitelisted hotkeys
        """
        bt.logging.trace(f"Checking whitelist for hotkey: {hotkey}")
        if isinstance(hotkey, bool) or not isinstance(hotkey, str):
            bt.logging.trace(f"Invalid hotkey type: {type(hotkey)}")
            return False

        whitelisted_hotkeys = []

        if hotkey in whitelisted_hotkeys:
            bt.logging.trace(f"Hotkey {hotkey} is whitelisted")
            return True

        bt.logging.trace(f"Hotkey {hotkey} is not whitelisted")
        return False

    def blacklist(self, synapse: GameData) -> Tuple[bool, str]:
        """
        Check if a synapse should be blacklisted.

        Args:
            synapse (GameData): The incoming synapse.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean (True if blacklisted) and a reason string.

        Behavior:
            - Checks if the sender is whitelisted
            - Verifies if the sender is a known validator
            - Checks if the validator has sufficient stake
        """
        bt.logging.trace(f"Checking blacklist for synapse from {synapse.dendrite.hotkey}")
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
        """
        Calculate the priority of a synapse based on the sender's stake.

        Args:
            synapse (GameData): The incoming synapse.

        Returns:
            float: The priority value for the synapse.

        Behavior:
            - Checks if the sender is whitelisted
            - If not whitelisted, calculates priority based on stake
        """
        bt.logging.trace(f"Calculating priority for synapse from {synapse.dendrite.hotkey}")
        if self.check_whitelist(hotkey=synapse.dendrite.hotkey):
            bt.logging.trace(f"Whitelisted hotkey {synapse.dendrite.hotkey}, returning max priority")
            return 10000000.0

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = float(self.metagraph.S[uid])

        bt.logging.trace(f"Prioritized: {synapse.dendrite.hotkey} (UID: {uid} - Stake: {stake})")
        return stake

    def ensure_db_directory_exists(self):
        db_dir = os.path.dirname(self.args.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def initialize_database(self):
        bt.logging.debug(f"Initializing database at {self.args.db_path}")
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS predictions (
                                   predictionID TEXT PRIMARY KEY, 
                                   teamGameID TEXT, 
                                   minerID TEXT, 
                                   predictionDate TEXT, 
                                   predictedOutcome TEXT,
                                   teamA TEXT,
                                   teamB TEXT,
                                   wager REAL,
                                   teamAodds REAL,
                                   teamBodds REAL,
                                   tieOdds REAL,
                                   canOverwrite BOOLEAN,
                                   outcome TEXT
                                   )"""
                )
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS games (
                                   gameID TEXT PRIMARY KEY, 
                                   teamA TEXT,
                                   teamAodds REAL,
                                   teamB TEXT,
                                   teamBodds REAL,
                                   sport TEXT, 
                                   league TEXT, 
                                   externalID TEXT, 
                                   createDate TEXT, 
                                   lastUpdateDate TEXT, 
                                   eventStartDate TEXT, 
                                   active INTEGER, 
                                   outcome TEXT,
                                   tieOdds REAL,
                                   canTie BOOLEAN
                                   )"""
                )
        except sqlite3.Error as e:
            bt.logging.error(f"Failed to initialize local database: {e}")
            raise Exception("Failed to initialize local database")