from argparse import ArgumentParser
import bittensor as bt
import json
from typing import Tuple
import sqlite3
import os
import sys
import torch
from copy import deepcopy
import copy
from datetime import datetime, timedelta, timezone
from bettensor.protocol import TeamGamePrediction
import uuid
from pathlib import Path
from os import path, rename
import requests
import time
from dotenv import load_dotenv
import os
import asyncio
import concurrent.futures
import math
import numpy as np
import torch
from bettensor.utils.weights_functions import WeightSetter
from bettensor.utils.api_client import APIClient

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Get the grandparent directory
grandparent_dir = os.path.dirname(parent_dir)

# Get the great grandparent directory
great_grandparent_dir = os.path.dirname(grandparent_dir)

# Add parent, grandparent, and great grandparent directories to sys.path
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(great_grandparent_dir)
from base.neuron import BaseNeuron
from dotenv import load_dotenv


class BettensorValidator(BaseNeuron):
    default_db_path = "data/validator.db"

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser, profile="validator")
        parser.add_argument(
            "--db",
            type=str,
            default=self.default_db_path,
            help="Path to the validator database",
        )

        # Check if the arguments are already defined before adding them
        if not any(arg.dest == 'subtensor.network' for arg in parser._actions):
            parser.add_argument('--subtensor.network', type=str, help="The subtensor network to connect to")
        if not any(arg.dest == 'netuid' for arg in parser._actions):
            parser.add_argument('--netuid', type=int, help="The network UID")
        if not any(arg.dest == 'wallet.name' for arg in parser._actions):
            parser.add_argument('--wallet.name', type=str, help="The name of the wallet to use")
        if not any(arg.dest == 'wallet.hotkey' for arg in parser._actions):
            parser.add_argument('--wallet.hotkey', type=str, help="The hotkey of the wallet to use")
        if not any(arg.dest == 'logging.trace' for arg in parser._actions):
            parser.add_argument('--logging.trace', action='store_true', help="Enable trace logging")
        if not any(arg.dest == 'logging.debug' for arg in parser._actions):
            parser.add_argument('--logging.debug', action='store_true', help="Enable debug logging")
        if not any(arg.dest == 'logging.info' for arg in parser._actions):
            parser.add_argument('--logging.info', action='store_true', help="Enable info logging")
        if not any(arg.dest == 'subtensor.chain_endpoint' for arg in parser._actions):
            parser.add_argument('--subtensor.chain_endpoint', type=str, help="subtensor endpoint")

        args = parser.parse_args()

        self.timeout = 12
        self.neuron_config = None
        self.wallet = None
        self.dendrite = None
        self.metagraph = None
        self.scores = None
        self.hotkeys = None
        self.subtensor = None
        self.miner_responses = None
        self.max_targets = None
        self.target_group = None
        self.blacklisted_miner_hotkeys = None
        self.load_validator_state = None
        self.data_entry = None
        self.uid = None
        self.last_stats_update = datetime.now(timezone.utc).date() - timedelta(days=1)
        self.loop = asyncio.get_event_loop()
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='asyncio')
        self.axon_port = getattr(args, 'axon.port', None) 
        self.db_path = "data/validator.db"
        self.api_hosts = {
            "baseball": "api-baseball.p.rapidapi.com",
            "soccer": "api-football-v1.p.rapidapi.com",
        }

        load_dotenv()  # take environment variables from .env.
        self.rapid_api_key = os.getenv("RAPID_API_KEY")
        self.api_client = APIClient(self.rapid_api_key)

        

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

    async def initialize_connection(self):
        if self.subtensor is None:
            try:
                self.subtensor = bt.subtensor(config=self.neuron_config)
                bt.logging.info(f"Connected to {self.neuron_config.subtensor.network} network")
            except Exception as e:
                bt.logging.error(f"Failed to initialize subtensor: {str(e)}")
                self.subtensor = None
        return self.subtensor

    def print_chain_endpoint(self):
        if self.subtensor:
            bt.logging.info(f"Current chain endpoint: {self.subtensor.chain_endpoint}")
        else:
            bt.logging.info("Subtensor is not initialized yet.")

    async def get_subtensor(self):
        if self.subtensor is None:
            self.subtensor = await self.initialize_connection()
        return self.subtensor

    async def sync_metagraph(self):
        subtensor = await self.get_subtensor()
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

        # read command line arguments and perform actions based on them
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
            self.db_path = args.db
        else:
            # setup initial scoring weights
            self.init_default_scores()
            self.max_targets = 256
            self.db_path = self.default_db_path

        self.target_group = 0

        # self.miner_stats = MinerStatsHandler(self.db_path, "validator")
        self.create_table()

        self.weight_setter = WeightSetter(
            metagraph=self.metagraph,
            wallet=self.wallet,
            subtensor=self.subtensor,
            neuron_config=self.neuron_config,
            loop=self.loop,
            thread_executor=self.thread_executor,
            db_path=self.db_path
        )

        self.weight_setter.update_all_daily_stats()
        return True

    def _parse_args(self, parser):
        """parses the command line arguments"""
        return parser.parse_args()

    def calculate_total_wager(self, cursor, minerId, event_start_date, exclude_id=None):
        """calculates the total wager for a given miner and event start date"""
        query = """
            SELECT p.wager 
            FROM predictions p
            JOIN game_data g ON p.teamGameId = g.id
            WHERE p.minerId = ? AND DATE(g.eventStartDate) = DATE(?)
        """
        params = (minerId, event_start_date)

        if exclude_id:
            query += " AND p.teamGameId != ?"
            params += (exclude_id,)

        cursor.execute(query, params)
        wagers = cursor.fetchall()
        total_wager = sum([w[0] for w in wagers])

        return total_wager

    def validator_validation(self, metagraph, wallet, subtensor) -> bool:
        """this method validates the validator has registered correctly"""
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"your validator: {wallet} is not registered to chain connection: {subtensor}. run btcli register and try again"
            )
            return False

        return True

    def insert_predictions(self, processed_uids, predictions):
        """
        Inserts new predictions into the database

        Args:
        processed_uids: list of uids that have been processed
        predictions: a dictionary with uids as keys and TeamGamePrediction objects as values
        """
        conn = self.connect_db()
        cursor = conn.cursor()
        current_time = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        # Get today's date in UTC
        today_utc = datetime.now(timezone.utc).date().isoformat()

        for uid, prediction_dict in predictions.items():
            for predictionID, res in prediction_dict.items():
                if int(uid) not in processed_uids:
                    bt.logging.info(f"UID {uid} not processed, skipping")
                    continue

                # Get today's date in UTC
                today_utc = datetime.now(timezone.utc).isoformat()

                hotkey = self.metagraph.hotkeys[int(uid)]
                predictionID = res.predictionID
                teamGameID = res.teamGameID
                minerId = hotkey
                predictionDate = today_utc
                predictedOutcome = res.predictedOutcome
                wager = res.wager

                # Check if the predictionID already exists
                cursor.execute(
                    "SELECT COUNT(*) FROM predictions WHERE predictionID = ?",
                    (predictionID,),
                )
                if cursor.fetchone()[0] > 0:
                    bt.logging.debug(
                        f"Prediction {predictionID} already exists, skipping."
                    )
                    continue

                query = "SELECT sport, league, eventStartDate, teamA, teamB, teamAodds, teamBodds, tieOdds, outcome FROM game_data WHERE externalId = ?"
                cursor.execute(query, (teamGameID,))
                result = cursor.fetchone()

                if not result:
                    continue

                (
                    sport,
                    league,
                    event_start_date,
                    teamA,
                    teamB,
                    teamAodds,
                    teamBodds,
                    tieOdds,
                    outcome,
                ) = result

                # Convert predictedOutcome to numeric value
                if predictedOutcome == teamA:
                    predictedOutcome = 0
                elif predictedOutcome == teamB:
                    predictedOutcome = 1
                elif predictedOutcome.lower() == "tie":
                    predictedOutcome = 2
                else:
                    bt.logging.debug(
                        f"Invalid predictedOutcome: {predictedOutcome}. Skipping this prediction."
                    )
                    continue

                # Check if the game has already started
                if current_time >= event_start_date:
                    bt.logging.debug(
                        f"Prediction not inserted: game {teamGameID} has already started."
                    )
                    continue

                # Calculate total wager for the date
                cursor.execute(
                    """
                    SELECT SUM(wager) FROM predictions
                    WHERE minerID = ? AND DATE(predictionDate) = DATE(?)
                """,
                    (minerId, predictionDate),
                )
                total_wager = cursor.fetchone()[0] or 0
                total_wager += wager

                if total_wager > 1000:
                    bt.logging.debug(
                        f"Total wager for the date exceeds $1000. Skipping this prediction."
                    )
                    continue

                # Insert new prediction
                cursor.execute(
                    """
                    INSERT INTO predictions (predictionID, teamGameID, minerID, predictionDate, predictedOutcome, teamA, teamB, wager, teamAodds, teamBodds, tieOdds, canOverwrite, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        predictionID,
                        teamGameID,
                        minerId,
                        predictionDate,
                        predictedOutcome,
                        teamA,
                        teamB,
                        wager,
                        teamAodds,
                        teamBodds,
                        tieOdds,
                        False,
                        outcome,
                    ),
                )

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    def connect_db(self):
        """connects to the sqlite database"""
        return sqlite3.connect(self.db_path)

    def create_table(self):
        """creates the predictions table if it doesn't exist"""
        conn = self.connect_db()
        c = conn.cursor()

        # create table if it doesn't exist
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS predictions (
            predictionID TEXT,
            teamGameID TEXT,
            minerId TEXT,
            predictionDate TEXT,
            predictedOutcome TEXT,
            teamA TEXT,
            teamB TEXT,
            wager REAL,
            teamAodds REAL,
            teamBodds REAL,
            tieOdds REAL,
            canOverwrite BOOLEAN,
            outcome TEXT,
            sent_to_site INTEGER DEFAULT 0
        )
        """
        )

        # commit the changes and close the connection
        conn.commit()
        conn.close()

    def process_prediction(
        self, processed_uids: torch.tensor, predictions: list
    ) -> list:
        """
        processes responses received by miners

        Args:
            processed_uids: list of uids that have been processed
            predictions: list of deserialized synapses
        """

        predictions_dict = {}

        for synapse in predictions:
            # ensure synapse has at least 3 elements
            if len(synapse) >= 3:
                prediction_data = synapse[0]
                prediction_dict: TeamGamePrediction = synapse[1]
                metadata = synapse[2]

                if metadata and hasattr(metadata, "neuron_uid"):
                    uid = metadata.neuron_uid

                    # ensure prediction_dict is not none before adding it to predictions_dict
                    if prediction_dict is not None and any(prediction_dict.values()):
                        predictions_dict[uid] = prediction_dict
                    else:
                        bt.logging.trace(
                            f"prediction from miner {uid} is none and will be skipped."
                        )
                else:
                    bt.logging.warning(
                        "metadata is missing or does not contain neuron_uid."
                    )
            else:
                bt.logging.warning(
                    "synapse data is incomplete or not in the expected format."
                )

        self.create_table()
        self.insert_predictions(processed_uids, predictions_dict)

    def add_new_miners(self):
        """
        adds new miners to the database, if there are new hotkeys in the metagraph
        """
        if self.hotkeys:
            uids_with_stake = self.metagraph.total_stake >= 0.0
            for i, hotkey in enumerate(self.metagraph.hotkeys):
                if (hotkey not in self.hotkeys) and (i not in uids_with_stake):
                    coldkey = self.metagraph.coldkeys[i]

                    if self.miner_stats.init_miner_row(hotkey, coldkey, i):
                        bt.logging.info(f"added new miner to the database: {hotkey}")
                    else:
                        bt.logging.error(
                            f"failed to add new miner to the database: {hotkey}"
                        )

    def check_hotkeys(self):
        """checks if some hotkeys have been replaced in the metagraph"""
        if self.hotkeys:
            # check if known state len matches with current metagraph hotkey lengt
            if len(self.hotkeys) == len(self.metagraph.hotkeys):
                current_hotkeys = self.metagraph.hotkeys
                for i, hotkey in enumerate(current_hotkeys):
                    if self.hotkeys[i] != hotkey:
                        bt.logging.debug(
                            f"index '{i}' has mismatching hotkey. old hotkey: '{self.hotkeys[i]}', new hotkey: '{hotkey}. resetting score to 0.0"
                        )
                        bt.logging.debug(f"score before reset: {self.scores[i]}")
                        self.scores[i] = 0.0
                        bt.logging.debug(f"score after reset: {self.scores[i]}")
            else:
                # TODO: Here, instead of resetting to default scores, we should just 
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
            metagraph_S_tensor = torch.from_numpy(self.metagraph.S).float()
            self.scores = torch.zeros_like(metagraph_S_tensor, dtype=torch.float32)
        
        bt.logging.info(f"validation weights have been initialized: {self.scores}")

    def evaluate_miner(self, minerId):
        """evaluates the performance of a miner

        Args:
            minerId: id of the miner to evaluate
        """
        # fetch data from predictions table for the specified minerId
        cursor.execute(
            """
        SELECT predictions.id, predictions.teamGameId, predictions.predictedOutcome, teamGame.outcome
        FROM predictions
        JOIN teamGame ON predictions.teamGameId = teamGame.id
        WHERE predictions.minerId = ?
        """,
            (miner_id,),
        )

        # update the predictionCorrect column based on the comparison
        for row in cursor.fetchall():
            prediction_id, team_game_id, predicted_outcome, actual_outcome = row
            prediction_correct = 1 if predicted_outcome == actual_outcome else 0
            cursor.execute(
                """
            UPDATE predictions
            SET predictionCorrect = ?
            WHERE id = ?
            """,
                (prediction_correct, prediction_id),
            )

        # commit the changes
        conn.commit()
        conn.close()

    def save_state(self):
        """saves the state of the validator to a file"""
        bt.logging.info("saving validator state")

        # save the state of the validator to file
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
                "last_updated_block": self.last_updated_block,
                "blacklisted_miner_hotkeys": self.blacklisted_miner_hotkeys,
            },
            self.base_path + "/state.pt",
        )

        bt.logging.debug(
            f"saved the following state to a file: step: {self.step}, scores: {self.scores}, hotkeys: {self.hotkeys}, last_updated_block: {self.last_updated_block}, blacklisted_miner_hotkeys: {self.blacklisted_miner_hotkeys}"
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
        """loads the state of the validator from a file"""

        # load the state of the validator from file
        state_path = self.base_path + "/state.pt"
        if path.exists(state_path):
            try:
                bt.logging.info("loading validator state")
                state = torch.load(state_path)
                bt.logging.debug(f"loaded the following state from file: {state}")
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.last_updated_block = state["last_updated_block"]
                if "blacklisted_miner_hotkeys" in state.keys():
                    self.blacklisted_miner_hotkeys = state["blacklisted_miner_hotkeys"]

                bt.logging.info(f"scores loaded from saved file: {self.scores}")
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
        blacklist_file = f"{self.base_path}/miner_blacklist.json"
        if Path(blacklist_file).is_file():
            # load the contents of the local blacklist
            bt.logging.trace(f"reading local blacklist file: {blacklist_file}")
            try:
                with open(blacklist_file, "r", encoding="utf-8") as file:
                    file_content = file.read()

                miner_blacklist = json.loads(file_content)
                if validate_miner_blacklist(miner_blacklist):
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

    def get_uids_to_query(self, all_axons) -> list:
        """returns the list of uids to query"""

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
        bt.logging.trace(f"uids with 0.0.0.0 as an ip address: {invalid_uids}")

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

    def update_game_outcome(self, game_id, numeric_outcome):
        """updates the outcome of a game in the database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE game_data SET outcome = ?, active = 0 WHERE externalId = ?",
                (numeric_outcome, game_id),
            )
            if cursor.rowcount == 0:
                bt.logging.trace(f"No game updated for externalId {game_id}")
            else:
                bt.logging.trace(
                    f"Updated game {game_id} with outcome: {numeric_outcome}"
                )
            conn.commit()
        except Exception as e:
            bt.logging.trace(f"Error updating game outcome: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_recent_games(self):
        """retrieves recent games from the database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        two_days_ago = (
            datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=48)
        ).isoformat()
        cursor.execute(
            "SELECT id, teamA, teamB, externalId FROM game_data WHERE eventStartDate >= ? AND outcome = 'Unfinished'",
            (two_days_ago,),
        )
        return cursor.fetchall()

    def determine_winner(self, game_info):
        game_id, teamA, teamB, externalId = game_info

        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT sport FROM game_data WHERE externalId = ?", (externalId,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            bt.logging.error(f"No game found with externalId {externalId}")
            return

        sport = result[0]

        if sport == "baseball":
            game_data = self.api_client.get_baseball_game(str(externalId))
        elif sport == "soccer":
            game_data = self.api_client.get_soccer_game(str(externalId))
        else:
            bt.logging.error(f"Unsupported sport: {sport}")
            return

        if not game_data:
            return

        game_response = game_data.get("response", [])[0]

        if sport == "baseball":
            status = game_response["status"]["long"]
            if status != "Finished":
                bt.logging.trace(f"Game {externalId} is not finished yet. Current status: {status}")
                return

            home_score = game_response["scores"]["home"]["total"]
            away_score = game_response["scores"]["away"]["total"]
        elif sport == "soccer":
            status = game_response["fixture"]["status"]["long"]
            if status not in ["Match Finished", "Match Finished After Extra Time", "Match Finished After Penalties"]:
                bt.logging.trace(f"Game {externalId} is not finished yet. Current status: {status}")
                return

            home_score = game_response["goals"]["home"]
            away_score = game_response["goals"]["away"]

            # Ensure home_score and away_score are not None
            if home_score is None or away_score is None:
                bt.logging.trace(f"Score data is incomplete for game {externalId}")
                return

            if home_score > away_score:
                numeric_outcome = 0
            elif away_score > home_score:
                numeric_outcome = 1
            else:
                numeric_outcome = 2

            bt.logging.trace(f"Game {externalId} result: {teamA} {home_score} - {away_score} {teamB}")
            bt.logging.trace(f"Numeric outcome: {numeric_outcome}")

            self.update_game_outcome(externalId, numeric_outcome)
        else:
            bt.logging.error(f"Failed to fetch game data for {externalId}. Status code: {response.status_code}")

    def update_recent_games(self):
        """Updates the outcomes of recent games and corresponding predictions"""
        recent_games = self.get_recent_games()

        for game_info in recent_games:
            game_id, teamA, teamB, externalId = game_info

            self.determine_winner(game_info)

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    conn = self.connect_db()
                    cursor = conn.cursor()

                    try:
                        # Fetch the updated outcome from game_data
                        cursor.execute(
                            "SELECT outcome FROM game_data WHERE externalId = ?",
                            (externalId,),
                        )
                        result = cursor.fetchone()

                        if result is None:
                            bt.logging.warning(
                                f"No game found with externalId {externalId}"
                            )
                            break

                        new_outcome = result[0]

                        if new_outcome == "Unfinished":
                            break

                        # Update predictions table where outcome is 'Unfinished' and matches teamGameID
                        cursor.execute(
                            """
                            UPDATE predictions
                            SET outcome = ?
                            WHERE teamGameID = ? AND outcome = 'Unfinished'
                        """,
                            (new_outcome, externalId),
                        )

                        conn.commit()
                        bt.logging.info(
                            f"Updated predictions for game {externalId} with outcome {new_outcome}"
                        )
                        break  # If successful, break the retry loop

                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < max_retries - 1:
                            bt.logging.warning(
                                f"Database locked, retrying in 1 second... (Attempt {attempt + 1})"
                            )
                            time.sleep(1)
                        else:
                            bt.logging.error(
                                f"Error updating predictions for game {externalId}: {e}"
                            )
                            break
                    except Exception as e:
                        bt.logging.error(
                            f"Error updating predictions for game {externalId}: {e}"
                        )
                        break
                    finally:
                        conn.close()

                except Exception as e:
                    bt.logging.error(f"Error connecting to database: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        break

        bt.logging.info("Recent games and predictions update process completed")

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    async def set_weights(self):
        await self.weight_setter.set_weights(self.db_path)
