import asyncio
import os
import sys
import copy
import json
import math
import time
import traceback
import uuid
import torch
import numpy as np
import requests
import bittensor as bt
import concurrent.futures
from os import path, rename
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv
from argparse import ArgumentParser
from typing import Dict, Tuple
from datetime import datetime, timedelta, timezone
from bettensor.base.neuron import BaseNeuron
from bettensor.protocol import TeamGamePrediction
from bettensor.validator.utils.io.website_handler import WebsiteHandler
from bettensor.validator.utils.scoring.scoring import ScoringSystem
from .utils.scoring.entropy_system import EntropySystem
from bettensor.validator.utils.io.sports_data import SportsData
from bettensor.validator.utils.scoring.weights_functions import WeightSetter
from bettensor.validator.utils.database.database_manager import DatabaseManager
from bettensor.validator.utils.io.miner_data import MinerDataMixin
from bettensor.validator.utils.io.bettensor_api_client import BettensorAPIClient
from bettensor.validator.utils.io.base_api_client import BaseAPIClient

DEFAULT_DB_PATH = "./bettensor/validator/state/validator.db"


class BettensorValidator(BaseNeuron, MinerDataMixin):
    """
    Bettensor Validator Class, Extends the BaseNeuron Class and MinerDataMixin Class

    Contains top-level methods for validator operations.
    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser, profile="validator")
        MinerDataMixin.__init__(self)  # Explicitly initialize the mixin
        parser.add_argument(
            "--db",
            type=str,
            default=DEFAULT_DB_PATH,
            help="Path to the validator database",
        )
        # Check if the arguments are already defined before adding them
        if not any(arg.dest == "subtensor.network" for arg in parser._actions):
            parser.add_argument(
                "--subtensor.network",
                type=str,
                help="The subtensor network to connect to",
            )
        if not any(arg.dest == "netuid" for arg in parser._actions):
            parser.add_argument("--netuid", type=int, help="The network UID")
        if not any(arg.dest == "wallet.name" for arg in parser._actions):
            parser.add_argument(
                "--wallet.name", type=str, help="The name of the wallet to use"
            )
        if not any(arg.dest == "wallet.hotkey" for arg in parser._actions):
            parser.add_argument(
                "--wallet.hotkey", type=str, help="The hotkey of the wallet to use"
            )
        if not any(arg.dest == "logging.trace" for arg in parser._actions):
            parser.add_argument(
                "--logging.trace", action="store_true", help="Enable trace logging"
            )
        if not any(arg.dest == "logging.debug" for arg in parser._actions):
            parser.add_argument(
                "--logging.debug", action="store_true", help="Enable debug logging"
            )
        if not any(arg.dest == "logging.info" for arg in parser._actions):
            parser.add_argument(
                "--logging.info", action="store_true", help="Enable info logging"
            )
        if not any(arg.dest == "subtensor.chain_endpoint" for arg in parser._actions):
            parser.add_argument(
                "--subtensor.chain_endpoint", type=str, help="subtensor endpoint"
            )

        args = parser.parse_args()

        self.timeout = 12 
        self.neuron_config = None
        self.wallet = None
        self.dendrite = None
        self.metagraph = None
        self.scores = None
        self.hotkeys = None
        self.subtensor = None
        self.axon_port = getattr(args, "axon.port", None)
        self.base_path = "./bettensor/validator/"
        self.max_targets = None
        self.target_group = None
        self.blacklisted_miner_hotkeys = None
        self.load_validator_state = None
        self.data_entry = None
        self.uid = None
        self.miner_responses = None
        self.db_path = DEFAULT_DB_PATH
        self.last_stats_update = datetime.now(timezone.utc).date() - timedelta(days=1)
        self.last_api_call = datetime.now(timezone.utc) - timedelta(
            days=15
        )  # set to 15 days ago to ensure all games are fetched
        self.is_primary = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.determine_max_workers())

        self.last_queried_block = 0
        self.last_sent_data_to_website = 0
        self.last_scoring_block = 0
        self.last_set_weights_block = 0
        self.operation_lock = asyncio.Lock()


    def determine_max_workers(self):
        num_cores = os.cpu_count()
        if num_cores <= 4:
            return num_cores - 1
        else:
            return num_cores - 2

    def apply_config(self, bt_classes) -> bool:
        """applies the configuration to specified bittensor classes"""
        try:
            self.neuron_config = self.config(bt_classes=bt_classes)
        except AttributeError as e:
            bt.logging.error(f"unable to apply validator configuration: {e}")
            raise AttributeError from e
        except OSError as e:
            bt.logging.error(f"unable to create logging directory: {e}")
            raise OSError from e

        return True

    def initialize_connection(self):
        max_retries = 5
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                self.subtensor = bt.subtensor(config=self.neuron_config)
                bt.logging.info(f"Connected to {self.neuron_config.subtensor.network} network")
                return self.subtensor
            except Exception as e:
                bt.logging.error(f"Failed to initialize subtensor (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    bt.logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    bt.logging.error("Max retries reached. Unable to initialize subtensor.")
                    self.subtensor = None
        return self.subtensor

    def print_chain_endpoint(self):
        if self.subtensor:
            bt.logging.info(f"Current chain endpoint: {self.subtensor.chain_endpoint}")
        else:
            bt.logging.info("Subtensor is not initialized yet.")

    def get_subtensor(self):
        if self.subtensor is None:
            self.subtensor = self.initialize_connection()
        return self.subtensor

    def sync_metagraph(self):
        subtensor = self.get_subtensor()
        self.metagraph.sync(subtensor=subtensor, lite=True)
        return self.metagraph

    def check_vali_reg(self, metagraph, wallet, subtensor) -> bool:
        """validates the validator has registered correctly"""
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"your validator: {wallet} is not registered to chain connection: {subtensor}. run btcli register and try again"
            )
            return False

        return True

    def setup_bittensor_objects(
        self, neuron_config
    ) -> Tuple[bt.wallet, bt.subtensor, bt.dendrite, bt.metagraph]:
        """sets up the bittensor objects"""
        try:
            wallet = bt.wallet(config=neuron_config)
            subtensor = bt.subtensor(config=neuron_config)
            dendrite = bt.dendrite(wallet=wallet)
            metagraph = subtensor.metagraph(neuron_config.netuid)
        except AttributeError as e:
            bt.logging.error(f"unable to setup bittensor objects: {e}")
            raise AttributeError from e

        self.hotkeys = copy.deepcopy(metagraph.hotkeys)

        return wallet, subtensor, dendrite, metagraph

    def serve_axon(self):
        """Serve the axon to the network"""
        bt.logging.info("Serving axon...")

        self.axon = bt.axon(wallet=self.wallet)

        self.axon.serve(netuid=self.neuron_config.netuid, subtensor=self.subtensor)

    def initialize_neuron(self) -> bool:
        """initializes the neuron

        Args:
            none

        Returns:
            bool:
                a boolean value indicating success/failure of the initialization
        Raises:
            AttributeError:
                AttributeError is raised if the neuron initialization failed
            IndexError:
                IndexError is raised if the hotkey cannot be found from the metagraph
        """
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        bt.logging.info(
            f"initializing validator for subnet: {self.neuron_config.netuid} on network: {self.neuron_config.subtensor.chain_endpoint} with config: {self.neuron_config}"
        )

        # setup the bittensor objects
        wallet, subtensor, dendrite, metagraph = self.setup_bittensor_objects(
            self.neuron_config
        )

        bt.logging.info(
            f"bittensor objects initialized:\nmetagraph: {metagraph}\nsubtensor: {subtensor}\nwallet: {wallet}"
        )

        # validate that the validator has registered to the metagraph correctly
        if not self.validator_validation(metagraph, wallet, subtensor):
            raise IndexError("unable to find validator key from metagraph")

        # get the unique identity (uid) from the network
        validator_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

        self.uid = validator_uid
        bt.logging.info(f"validator is running with uid: {validator_uid}")

        self.wallet = wallet
        self.subtensor = subtensor
        self.dendrite = dendrite
        self.metagraph = metagraph

        if self.metagraph is not None:
            self.scores = torch.zeros(len(self.metagraph.uids), dtype=torch.float32)

        #
        args = self._parse_args(parser=self.parser)

        if args:
            if args.load_state == "False":
                self.load_validator_state = False
            else:
                self.load_validator_state = True

            if self.load_validator_state:
                self.load_state()
            else:
                self.init_default_scores()

            if args.max_targets:
                self.max_targets = args.max_targets
            else:
                self.max_targets = 256

        else:
            # setup initial scoring weights
            self.init_default_scores()
            self.max_targets = 256

        self.db_path = args.db
        self.target_group = 0

        self.db_manager = DatabaseManager(self.db_path)

        self.scoring_system = ScoringSystem(
            self.db_manager,
            num_miners=256,
            max_days=45,
            reference_date=datetime.now(timezone.utc).date(),
            validator=self,
        )

        ############## Setup Validator Components ##############
        self.api_client = BettensorAPIClient(self.db_manager)

        self.sports_data = SportsData(
            db_manager=self.db_manager,
            api_client=self.api_client,
            entropy_system=self.scoring_system.entropy_system,
        )

        self.weight_setter = WeightSetter(
            metagraph=self.metagraph,
            wallet=self.wallet,
            subtensor=self.subtensor,
            neuron_config=self.neuron_config,
            db_path=self.db_path,
        )

        self.website_handler = WebsiteHandler(self)

        return True

    def _parse_args(self, parser):
        """parses the command line arguments"""
        return parser.parse_args()

    def validator_validation(self, metagraph, wallet, subtensor) -> bool:
        """this method validates the validator has registered correctly"""
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"your validator: {wallet} is not registered to chain connection: {subtensor}. run btcli register and try again"
            )
            return False

        return True

    def check_hotkeys(self):
        """checks if some hotkeys have been replaced in the metagraph"""
        bt.logging.info("Checking metagraph hotkeys for changes")
        if self.scores is None:
            if self.metagraph is not None:
                self.scores = torch.zeros(len(self.metagraph.uids), dtype=torch.float32)
            else:
                bt.logging.warning("Metagraph is None, unable to initialize scores")
                return
        if self.hotkeys:
            # check if known state len matches with current metagraph hotkey length
            if len(self.hotkeys) == len(self.metagraph.hotkeys):
                current_hotkeys = self.metagraph.hotkeys
                for i, hotkey in enumerate(current_hotkeys):
                    if self.hotkeys[i] != hotkey:
                        bt.logging.debug(
                            f"index '{i}' has mismatching hotkey. old hotkey: '{self.hotkeys[i]}', new hotkey: '{hotkey}. resetting score to 0.0"
                        )
                        self.scores[i] = 0.0
                        self.scoring_system.reset_miner(i)
                        self.save_state()
                        
            else:
                bt.logging.info(
                    f"init default scores because of state and metagraph hotkey length mismatch. expected: {len(self.metagraph.hotkeys)} had: {len(self.hotkeys)}"
                )
                self.init_default_scores()

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        else:
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def init_default_scores(self):
        """Initialize default scores for all miners in the network. This method is
        used to reset the scores in case of an internal error"""

        bt.logging.info("initiating validator with default scores for all miners")

        if self.metagraph is None or self.metagraph.S is None:
            bt.logging.error("Metagraph or metagraph.S is not initialized")
            self.scores = torch.zeros(1, dtype=torch.float32)
        else:
            # Convert numpy array to PyTorch tensor
            if isinstance(self.metagraph.S, np.ndarray):
                metagraph_S_tensor = torch.from_numpy(self.metagraph.S).float()
            elif isinstance(self.metagraph.S, torch.Tensor):
                metagraph_S_tensor = self.metagraph.S.float()
            else:
                bt.logging.error(
                    f"Unexpected type for metagraph.S: {type(self.metagraph.S)}"
                )
                metagraph_S_tensor = torch.zeros(
                    len(self.metagraph.hotkeys), dtype=torch.float32
                )
                self.scores = torch.zeros_like(metagraph_S_tensor, dtype=torch.float32)

        bt.logging.info(f"validation weights have been initialized: {self.scores}")

    def save_state(self):
        """saves the state of the validator to a file"""
        bt.logging.info("Saving validator state")

        bt.logging.info(f"Last api call, save_state: {self.last_api_call}")

        if isinstance(self.last_api_call, str):
            self.last_api_call = datetime.fromisoformat(self.last_api_call)

        bt.logging.info(f"Last api call, save_state: {self.last_api_call}")
        timestamp = self.last_api_call.timestamp()

        # save the state of the validator to file
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
                "last_updated_block": self.last_updated_block,
                "blacklisted_miner_hotkeys": self.blacklisted_miner_hotkeys,
                "last_api_call": timestamp,
            },
            self.base_path + "state/state.pt",
        )

        bt.logging.debug(
            f"Saved the following state to a file: step: {self.step}, scores: {self.scores}, hotkeys: {self.hotkeys}, "
            f"last_updated_block: {self.last_updated_block}, blacklisted_miner_hotkeys: {self.blacklisted_miner_hotkeys}, "
            f"last_api_call: {timestamp}"
        )

    def reset_validator_state(self, state_path):
        """inits the default validator state. should be invoked only
        when an exception occurs and the state needs to reset"""

        # rename current state file in case manual recovery is needed
        rename(
            state_path,
            f"{state_path}-{int(datetime.now().timestamp())}.autorecovery",
        )

        self.init_default_scores()
        self.step = 0
        self.last_updated_block = 0
        self.hotkeys = None
        self.blacklisted_miner_hotkeys = None

    def load_state(self):
        state_path = self.base_path + "state/state.pt"
        if path.exists(state_path):
            try:
                bt.logging.info("Loading validator state")
                state = torch.load(state_path)
                bt.logging.debug(f"Loaded the following state from file: {state}")
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.last_updated_block = state["last_updated_block"]
                if "blacklisted_miner_hotkeys" in state.keys():
                    self.blacklisted_miner_hotkeys = state["blacklisted_miner_hotkeys"]

                # Convert timestamps back to datetime
                last_api_call = state["last_api_call"]
                if last_api_call is None:
                    self.last_api_call = datetime.now(timezone.utc) - timedelta(
                        minutes=30
                    )
                else:
                    try:
                        self.last_api_call = datetime.fromtimestamp(
                            last_api_call, tz=timezone.utc
                        )

                    except (ValueError, TypeError, OverflowError) as e:
                        bt.logging.warning(
                            f"Invalid last_api_call timestamp: {last_api_call}. Using current time. Error: {e}"
                        )
                        self.last_api_call = datetime.now(timezone.utc)

            except Exception as e:
                bt.logging.error(
                    f"validator state reset because an exception occurred: {e}"
                )
                self.reset_validator_state(state_path=state_path)
        else:
            self.init_default_scores()

    def _get_local_miner_blacklist(self) -> list:
        """returns the blacklisted miners hotkeys from the local file"""

        # check if local blacklist exists
        blacklist_file = f"{self.base_path}state/miner_blacklist.json"
        if Path(blacklist_file).is_file():
            # load the contents of the local blacklist
            bt.logging.trace(f"reading local blacklist file: {blacklist_file}")
            try:
                with open(blacklist_file, "r", encoding="utf-8") as file:
                    file_content = file.read()

                miner_blacklist = json.loads(file_content)
                if self.validate_miner_blacklist(miner_blacklist):
                    bt.logging.trace(f"loaded miner blacklist: {miner_blacklist}")
                    return miner_blacklist

                bt.logging.trace(
                    f"loaded miner blacklist was formatted incorrectly or was empty: {miner_blacklist}"
                )
            except OSError as e:
                bt.logging.error(f"unable to read blacklist file: {e}")
            except json.JSONDecodeError as e:
                bt.logging.error(
                    f"unable to parse json from path: {blacklist_file} with error: {e}"
                )
        else:
            bt.logging.trace(f"no local miner blacklist file in path: {blacklist_file}")

        return []

    def validate_miner_blacklist(self, miner_blacklist) -> bool:
        """validates the miner blacklist. checks if the list is not empty and if all the hotkeys are in the metagraph"""
        blacklist_file = f"{self.base_path}/miner_blacklist.json"
        if not miner_blacklist:
            return False
        if not all(hotkey in self.metagraph.hotkeys for hotkey in miner_blacklist):
            # update the blacklist with the valid hotkeys
            valid_hotkeys = [
                hotkey for hotkey in miner_blacklist if hotkey in self.metagraph.hotkeys
            ]
            self.blacklisted_miner_hotkeys = valid_hotkeys
            # overwrite the old blacklist with the new blacklist
            with open(blacklist_file, "w", encoding="utf-8") as file:
                json.dump(valid_hotkeys, file)
        return True

    def get_uids_to_query(self, all_axons) -> list:
        """returns the list of uids to query"""

        # Define all_uids at the beginning
        all_uids = set(range(len(self.metagraph.hotkeys)))

        # get uids with a positive stake
        uids_with_stake = self.metagraph.total_stake >= 0.0
        bt.logging.trace(f"uids with a positive stake: {uids_with_stake}")

        # get uids with an ip address of 0.0.0.0
        invalid_uids = torch.tensor(
            [
                bool(value)
                for value in [
                    ip != "0.0.0.0"
                    for ip in [
                        self.metagraph.neurons[uid].axon_info.ip
                        for uid in self.metagraph.uids.tolist()
                    ]
                ]
            ],
            dtype=torch.bool,
        )

        # Append the validator's axon to invalid_uids
        invalid_uids[self.uid] = True

        bt.logging.trace(f"uids with 0.0.0.0 as an ip address or validator's axon: {invalid_uids}")

       

        # get uids that have their hotkey blacklisted
        blacklisted_uids = []
        if self.blacklisted_miner_hotkeys:
            for hotkey in self.blacklisted_miner_hotkeys:
                if hotkey in self.metagraph.hotkeys:
                    blacklisted_uids.append(self.metagraph.hotkeys.index(hotkey))
                else:
                    bt.logging.trace(
                        f"blacklisted hotkey {hotkey} was not found from metagraph"
                    )

            bt.logging.debug(f"blacklisted the following uids: {blacklisted_uids}")

        # convert blacklisted uids to tensor
        blacklisted_uids_tensor = torch.tensor(
            [uid not in blacklisted_uids for uid in self.metagraph.uids.tolist()],
            dtype=torch.bool,
        )

        bt.logging.trace(f"blacklisted uids: {blacklisted_uids_tensor}")

        # determine the uids to filter
        uids_to_filter = torch.logical_not(
            ~blacklisted_uids_tensor | ~invalid_uids | ~uids_with_stake
        )

        bt.logging.trace(f"uids to filter: {uids_to_filter}")

        # define uids to query
        uids_to_query = [
            axon
            for axon, keep_flag in zip(all_axons, uids_to_filter)
            if keep_flag.item()
        ]

        # define uids to filter
        final_axons_to_filter = [
            axon
            for axon, keep_flag in zip(all_axons, uids_to_filter)
            if not keep_flag.item()
        ]

        uids_not_to_query = [
            self.metagraph.hotkeys.index(axon.hotkey) for axon in final_axons_to_filter
        ]

        bt.logging.trace(f"final axons to filter: {final_axons_to_filter}")
        bt.logging.debug(f"filtered uids: {uids_not_to_query}")

        # reduce the number of simultaneous uids to query
        if self.max_targets < 256:
            start_idx = self.max_targets * self.target_group
            end_idx = min(
                len(uids_to_query), self.max_targets * (self.target_group + 1)
            )
            if start_idx == end_idx:
                return [], []
            if start_idx >= len(uids_to_query):
                raise IndexError(
                    "starting index for querying the miners is out-of-bounds"
                )

            if end_idx >= len(uids_to_query):
                end_idx = len(uids_to_query)
                self.target_group = 0
            else:
                self.target_group += 1

            bt.logging.debug(
                f"list indices for uids to query starting from: '{start_idx}' ending with: '{end_idx}'"
            )
            uids_to_query = uids_to_query[start_idx:end_idx]

        list_of_uids = [
            self.metagraph.hotkeys.index(axon.hotkey) for axon in uids_to_query
        ]

        list_of_hotkeys = [axon.hotkey for axon in uids_to_query]

        bt.logging.trace(f"sending query to the following hotkeys: {list_of_hotkeys}")

        return uids_to_query, list_of_uids, blacklisted_uids, uids_not_to_query

    def set_weights(self, scores):
        try:
            return self.weight_setter.set_weights(scores)
        except StopIteration:
            bt.logging.warning(
                "StopIteration encountered in set_weights. Handling gracefully."
            )
            return None

    def reset_scoring_system(self):
        """
        Resets the scoring/database system across all validators.
        """
        try:
            bt.logging.info("Resetting scoring system(deleting state files)...")
                        
            # delete state files (./bettensor/validator/state/state.pt, ./bettensor/validator/state/validator.db, ./bettensor/validator/state/entropy_system_state.json)
            for file in ["./bettensor/validator/state/state.pt", "./bettensor/validator/state/validator.db", "./bettensor/validator/state/entropy_system_state.json"]:
                if os.path.exists(file):
                    os.remove(file)
            
        except Exception as e:
            bt.logging.error(f"Error resetting scoring system: {e}")
            bt.logging.error(traceback.format_exc())
            raise