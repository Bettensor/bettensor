from argparse import ArgumentParser
from bettensor.utils.miner_stats import MinerStatsHandler
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

        self.timeout = 12
        self.neuron_config = None
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        self.scores = None
        self.hotkeys = None
        self.miner_responses = None
        self.max_targets = None
        self.target_group = None
        self.blacklisted_miner_hotkeys = None
        self.load_validator_state = None
        self.data_entry = None
        self.uid = None

        load_dotenv()  # take environment variables from .env.

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
        bt.logging.info(f"predictions: {predictions}")
        conn = self.connect_db()
        cursor = conn.cursor()
        current_time = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        for uid, prediction_dict in predictions.items():
            for predictionID, res in prediction_dict.items():
                if int(uid) not in processed_uids:
                    bt.logging.info(f"UID {uid} not processed, skipping")
                    continue

                hotkey = self.metagraph.hotkeys[int(uid)]
                predictionID = res.predictionID
                teamGameID = res.teamGameID
                minerId = hotkey
                predictionDate = res.predictionDate
                predictedOutcome = res.predictedOutcome
                wager = res.wager

                # Check if the predictionID already exists
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE predictionID = ?", (predictionID,))
                if cursor.fetchone()[0] > 0:
                    bt.logging.info(f"Prediction {predictionID} already exists, skipping.")
                    continue

                query = "SELECT sport, league, eventStartDate, teamA, teamB, teamAodds, teamBodds, tieOdds, outcome FROM game_data WHERE externalId = ?"
                cursor.execute(query, (teamGameID,))
                result = cursor.fetchone()

                if not result:
                    bt.logging.warning(f"No game data found for teamGameID: {teamGameID}. Skipping this prediction.")
                    continue

                sport, league, event_start_date, teamA, teamB, teamAodds, teamBodds, tieOdds, outcome = result

                # Convert predictedOutcome to numeric value
                if predictedOutcome == teamA:
                    predictedOutcome = 0
                elif predictedOutcome == teamB:
                    predictedOutcome = 1
                elif predictedOutcome.lower() == "tie":
                    predictedOutcome = 2
                else:
                    bt.logging.warning(f"Invalid predictedOutcome: {predictedOutcome}. Skipping this prediction.")
                    continue

                # Check if the game has already started
                if current_time >= event_start_date:
                    bt.logging.info(f"Prediction not inserted: game {teamGameID} has already started.")
                    continue

                # Calculate total wager for the date
                cursor.execute("""
                    SELECT SUM(wager) FROM predictions
                    WHERE minerID = ? AND DATE(predictionDate) = DATE(?)
                """, (minerId, predictionDate))
                total_wager = cursor.fetchone()[0] or 0
                total_wager += wager

                if total_wager > 1000:
                    bt.logging.warning(f"Total wager for the date exceeds $1000. Skipping this prediction.")
                    continue

                # Insert new prediction
                cursor.execute("""
                    INSERT INTO predictions (predictionID, teamGameID, minerID, predictionDate, predictedOutcome, teamA, teamB, wager, teamAodds, teamBodds, tieOdds, canOverwrite, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (predictionID, teamGameID, minerId, predictionDate, predictedOutcome, teamA, teamB, wager, teamAodds, teamBodds, tieOdds, False, outcome))

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
            outcome TEXT
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
                    bt.logging.info(f"processing prediction from miner: {uid}")
                    bt.logging.info(f"prediction: {prediction_dict}")
                    bt.logging.info(f"prediction type: {type(prediction_dict)}")

                    # ensure prediction_dict is not none before adding it to predictions_dict
                    if prediction_dict is not None:
                        predictions_dict[uid] = prediction_dict
                    else:
                        bt.logging.warning(
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
            # check if known state len matches with current metagraph hotkey length
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
                # init default scores
                bt.logging.info(
                    f"init default scores because of state and metagraph hotkey length mismatch. expected: {len(self.metagraph.hotkeys)} had: {len(self.hotkeys)}"
                )
                self.init_default_scores()

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        else:
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def init_default_scores(self) -> None:
        """validators without previous validation knowledge should start
        with default score of 0.0 for each uid. the method can also be
        used to reset the scores in case of an internal error"""

        bt.logging.info("initiating validator with default scores for each uid")
        self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
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

    def sync_metagraph(self, metagraph, subtensor):
        """syncs the metagraph"""

        bt.logging.debug(
            f"syncing metagraph: {self.metagraph} with subtensor: {self.subtensor}"
        )

        # sync the metagraph
        metagraph.sync(subtensor=subtensor)

        return metagraph

    # need func to set weights; dont think i should take fans?

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

    def update_game_outcome(self, game_id, outcome):
        """updates the outcome of a game in the database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE game_data SET outcome = ?, active = 1 WHERE id = ?",
            (outcome, game_id),
        )
        conn.commit()
        conn.close()

    def get_recent_games(self):
        """retrieves recent games from the database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        three_days_ago = datetime.utcnow().replace(
            tzinfo=timezone.utc
        ).isoformat() - timedelta(hours=72)
        three_days_ago_str = three_days_ago.isoformat()
        cursor.execute(
            "SELECT id, teamA, teamB, externalId FROM game_data WHERE eventStartDate >= ? AND active = 0",
            (three_days_ago_str,),
        )
        return cursor.fetchall()

    def determine_winner(self, game_info):
        """determines the winner of a game using an external api"""
        game_id, teamA, teamB, externalId = game_info

        url = "https://api-baseball.p.rapidapi.com/games"
        headers = {
            "x-rapidapi-host": "api-baseball.p.rapidapi.com",
            "x-rapidapi-key": "b416b1c26dmsh6f20cd13ee1f7ccp11cc1djsnf64975aaacde",
        }
        querystring = {"id": str(externalId)}

        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            data = response.json()
            game_response = data.get("response", [])[0]

            home_team = game_response["teams"]["home"]["name"]
            away_team = game_response["teams"]["away"]["name"]
            home_score = game_response["scores"]["home"]["total"]
            away_score = game_response["scores"]["away"]["total"]

            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    winner = teamA if teamA == home_team else teamB
                elif away_score > home_score:
                    winner = teamB if teamB == away_team else teamA
                else:
                    winner = "tie"

                self.update_game_outcome(game_id, winner)

    def update_recent_games(self):
        """updates the outcomes of recent games"""
        recent_games = self.get_recent_games()
        for game_info in recent_games:
            self.determine_winner(game_info)

    def set_weights(self):
        """sets the weights for the miners based on their performance in the last 48 hours"""
        # initialize the earnings tensor
        earnings = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

        # connect to the sqlite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # get the current timestamp
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        # calculate the timestamp for 48 hours ago
        forty_eight_hours_ago = now - timedelta(hours=48)

        # fetch the relevant data from game_data for the last 48 hours
        cursor.execute(
            "SELECT externalId, eventStartDate FROM game_data WHERE eventStartDate >= ?",
            (forty_eight_hours_ago.strftime("%Y-%m-%d %H:%M:%S"),)
        )
        game_data_rows = cursor.fetchall()

        # create a mapping from teamGameID to eventStartDate
        game_date_map = {row[0]: row[1] for row in game_data_rows}

        # fetch all the relevant data from predictions
        cursor.execute(
            "SELECT predictionID, teamGameID, minerId, predictedOutcome, outcome, teamA, teamB, wager, teamAodds, teamBodds FROM predictions"
        )
        prediction_rows = cursor.fetchall()

        # close the database connection
        conn.close()

        # process the data
        miner_performance = {}
        miner_id_to_index = {
            miner_id: idx for idx, miner_id in enumerate(self.metagraph.hotkeys)
        }

        for row in prediction_rows:
            prediction_id, team_game_id, miner_id, predicted_outcome, outcome, team_a, team_b, wager, team_a_odds, team_b_odds = row

            if team_game_id in game_date_map:
                event_date = datetime.strptime(game_date_map[team_game_id], "%Y-%m-%d %H:%M:%S")
                if event_date >= forty_eight_hours_ago:
                    if miner_id not in miner_performance:
                        miner_performance[miner_id] = 0.0

                    if predicted_outcome == outcome:
                        if predicted_outcome == team_a:
                            earned = wager * team_a_odds
                        elif predicted_outcome == team_b:
                            earned = wager * team_b_odds
                        else:
                            earned = 0  # in case there's some other outcome handling needed
                        miner_performance[miner_id] += earned

        # update the earnings tensor
        for miner_id, total_earned in miner_performance.items():
            if miner_id in miner_id_to_index:
                idx = miner_id_to_index[miner_id]
                earnings[idx] = total_earned

        # normalize the earnings tensor
        weights = torch.nn.functional.normalize(earnings, p=1.0, dim=0)

        # check stake and set weights
        uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        stake = float(self.metagraph.S[uid])
        if stake < 0.0:
            bt.logging.error("insufficient stake. failed in setting weights.")
        else:
            result = self.subtensor.set_weights(
                netuid=self.neuron_config.netuid,  # subnet to set weights on
                wallet=self.wallet,  # wallet to sign set weights using hotkey
                uids=self.metagraph.uids,  # uids of the miners to set weights for
                weights=weights,  # weights to set for the miners
                wait_for_inclusion=False,
            )
            if result:
                bt.logging.success("successfully set weights.")
            else:
                bt.logging.error("failed to set weights.")