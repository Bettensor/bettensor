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
from datetime import datetime
from bettensor.protocol import TeamGamePrediction
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
from pathlib import Path
from base.neuron import BaseNeuron
from os import path, rename
class BettensorValidator(BaseNeuron):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser, profile="validator")

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
        
    
    def apply_config(self, bt_classes) -> bool:
        """This method applies the configuration to specified bittensor classes"""
        try:
            self.neuron_config = self.config(bt_classes=bt_classes)
        except AttributeError as e:
            bt.logging.error(f"Unable to apply validator configuration: {e}")
            raise AttributeError from e
        except OSError as e:
            bt.logging.error(f"Unable to create logging directory: {e}")
            raise OSError from e

        return True

    def check_vali_reg(self, metagraph, wallet, subtensor) -> bool:
        """This method validates the validator has registered correctly"""
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again"
            )
            return False

        return True

    
    def setup_bittensor_objects(
        self, neuron_config
    ) -> Tuple[bt.wallet, bt.subtensor, bt.dendrite, bt.metagraph]:
        """Setups the bittensor objects"""
        try:
            wallet = bt.wallet(config=neuron_config)
            subtensor = bt.subtensor(config=neuron_config)
            dendrite = bt.dendrite(wallet=wallet)
            metagraph = subtensor.metagraph(neuron_config.netuid)
        except AttributeError as e:
            bt.logging.error(f"Unable to setup bittensor objects: {e}")
            raise AttributeError from e

        self.hotkeys = copy.deepcopy(metagraph.hotkeys)

        return wallet, subtensor, dendrite, metagraph


    def initialize_neuron(self) -> bool:
        """This function initializes the neuron.

        The setup function initializes the neuron by registering the
        configuration.

        Args:
            None

        Returns:
            Bool:
                A boolean value indicating success/failure of the initialization.
        Raises:
            AttributeError:
                AttributeError is raised if the neuron initialization failed
            IndexError:
                IndexError is raised if the hotkey cannot be found from the metagraph
        """
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        bt.logging.info(
            f"Initializing validator for subnet: {self.neuron_config.netuid} on network: {self.neuron_config.subtensor.chain_endpoint} with config: {self.neuron_config}"
        )

        # Setup the bittensor objects
        wallet, subtensor, dendrite, metagraph = self.setup_bittensor_objects(
            self.neuron_config
        )

        bt.logging.info(
            f"Bittensor objects initialized:\nMetagraph: {metagraph}\nSubtensor: {subtensor}\nWallet: {wallet}"
        )

        # Validate that the validator has registered to the metagraph correctly
        if not self.validator_validation(metagraph, wallet, subtensor):
            raise IndexError("Unable to find validator key from metagraph")

        # Get the unique identity (UID) from the network
        validator_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)


        self.uid = validator_uid
        bt.logging.info(f"Validator is running with UID: {validator_uid}")

        
        self.wallet = wallet
        self.subtensor = subtensor
        self.dendrite = dendrite
        self.metagraph = metagraph

        # Read command line arguments and perform actions based on them
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
            # Setup initial scoring weights
            self.init_default_scores()
            self.max_targets = 256

        self.target_group = 0

        self.miner_stats = MinerStatsHandler()

        return True

    def _parse_args(self, parser):
        return parser.parse_args()

    def calculate_total_wager(c, minerId, event_start_date, exclude_id=None):
        query = '''
            SELECT wager FROM predictions WHERE minerId = ? AND DATE(eventStartDate) = DATE(?)
        '''
        params = (minerId, event_start_date)
    
        if exclude_id:
            query += ' AND teamGameId != ?'
            params += (exclude_id,)

        c.execute(query, params)
        wagers = c.fetchall()
        total_wager = sum([w[0] for w in wagers])
    
        return total_wager

    def validator_validation(self, metagraph, wallet, subtensor) -> bool:
        """This method validates the validator has registered correctly"""
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again"
            )
            return False

        return True

    def insert_or_update_predictions(self, processed_uids, predictions):
        """
        Updates database with new predictions
        Args:
        processed_uids: list of uids that have been processed
        predictions: a dictionary with uids as keys and TeamGamePrediction objects as values

        """
        conn = self.connect_db()
        c = conn.cursor()
        current_time = datetime.now().isoformat()

        for uid, res in predictions.items():
            # TODO: nest another loop to iterate through all the predictions

            #ensure that prediction uid is in processed_uids 
            if uid not in processed_uids:
                #TODO : handle? Toss this prediction?
                continue
            
            # We need to pull some game data from the database to fill sport, league

            hotkey = self.metagraph.hotkeys[i]
            predictionID = res.predictionID
            teamGameId = res.teamGameID
            minerId = hotkey
            predictionDate = res.predictionDate
            predictedOutcome = res.predictedOutcome
            wager = res.wager
            query = "SELECT sport, league FROM game_data WHERE id = ?"

            # Execute the query
            cursor.execute(query, (team_game_id,))
            result = cursor.fetchone()

            # Check if the result is found and return it
            if result:
                sport, league = result
                return sport, league
            else:
                return None, None

            
            c.execute('''
                SELECT eventStartDate FROM game_data WHERE id = ?
            ''', (teamGameId))


            row = c.fetchone()
            if row:
                event_start_date = row[0]
                if current_time >= event_start_date:
                    print(f"Prediction not inserted/updated: Game {teamGameId} has already started.")
                    continue
            
            # Check if the prediction already exists; update prediction if it does
            c.execute('''
                SELECT id, wager FROM predictions WHERE predictionID = ?
            ''', (predictionID))
            
            existing_prediction = c.fetchone()
            
            if existing_prediction:
                # TODO: fix this function; does not properly check if wager > 1000, updates wager in weird way
                existing_id, existing_wager = existing_prediction
                total_wager = calculate_total_wager(c, minerId, event_start_date, exclude_id=teamGameId)
                
                # Add the new wager and subtract the existing one
                total_wager = total_wager - existing_wager + wager
                
                if total_wager > 1000:
                    print(f"Error: Total wager for the date exceeds $1000.")
                    continue
                
                # Update the existing prediction
                c.execute('''
                    UPDATE predictions
                    SET predictionDate = ?, predictedOutcome = ?, wager = ?
                    WHERE teamGameId = ? AND sport = ? AND minerId = ? AND league = ?
                ''', (predictionDate, predictedOutcome, wager, teamGameId, sport, minerId, league))
            else:
                total_wager = calculate_total_wager(c, minerId, event_start_date)
                
                # Add the new wager
                total_wager += wager
                
                if total_wager > 1000:
                    print(f"Error: Total wager for the date exceeds $1000.")
                    continue

                # Insert new prediction
                c.execute('''
                    INSERT INTO predictions (id, teamGameId, sport, minerId, league, predictionDate, predictedOutcome, wager)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (str(uuid4()), teamGameId, sport, minerId, league, predictionDate, predictedOutcome, wager))
        
        # Commit changes, close the connection
        conn.commit()
        conn.close()
        return sqlite3.connect('validator.db')

    
    def connect_db(self):
        return sqlite3.connect('validator.db')

    def create_table(self):
        conn = self.connect_db()
        c = conn.cursor()
        
        # Create table if it doesn't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            predictionID INTEGER PRIMARY KEY,
            teamGameID INTEGER,
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
        ''')
    
        # Commit the changes and close the connection
        conn.commit()
        conn.close()

    
    def process_prediction(self, processed_uids: torch.tensor, predictions:list) ->list:
        """
        Processes responses received by miners

        Args: 
            processed_uids: list of uids that have been processed
            predictions: list of deserialized synapses
        """ 
        
        predictions_dict = {}

        for synapse in predictions:
            prediction_dict : TeamGamePrediction = synapse[1]
            uid = synapse[2].neuron_uid
            bt.logging.info(f"Processing prediction from miner: {uid}")
            bt.logging.info(f"Prediction: {prediction_dict}")
            bt.logging.info(f"Prediction type: {type(prediction_dict)}")
            predictions_dict[uid] = prediction_dict       

        self.create_table()
        self.insert_or_update_predictions(processed_uids, predictions_dict)

    def add_new_miners(self):
        '''
        adds new miners to the database, if there are new hotkeys in the metagraph
        '''
        if self.hotkeys:
            uids_with_stake = self.metagraph.total_stake >= 0.0
            for i, hotkey in enumerate(self.metagraph.hotkeys):
                if (hotkey not in self.hotkeys) and (i not in uids_with_stake):
                    coldkey = self.metagraph.coldkeys[i]
                    


                    if self.miner_stats.init_miner_row(hotkey,coldkey,i):
                        bt.logging.info(f"Added new miner to the database: {hotkey}")
                    else:
                        bt.logging.error(f"Failed to add new miner to the database: {hotkey}")


    def check_hotkeys(self):
        """Checks if some hotkeys have been replaced in the metagraph"""
        if self.hotkeys:
            # Check if known state len matches with current metagraph hotkey length
            if len(self.hotkeys) == len(self.metagraph.hotkeys):
                current_hotkeys = self.metagraph.hotkeys
                for i, hotkey in enumerate(current_hotkeys):
                    if self.hotkeys[i] != hotkey:
                        bt.logging.debug(
                            f"Index '{i}' has mismatching hotkey. Old hotkey: '{self.hotkeys[i]}', new hotkey: '{hotkey}. Resetting score to 0.0"
                        )
                        bt.logging.debug(f"Score before reset: {self.scores[i]}")
                        self.scores[i] = 0.0
                        bt.logging.debug(f"Score after reset: {self.scores[i]}")
            else:
                # Init default scores
                bt.logging.info(
                    f"Init default scores because of state and metagraph hotkey length mismatch. Expected: {len(self.metagraph.hotkeys)} had: {len(self.hotkeys)}"
                )
                self.init_default_scores()

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        else:
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    
    def init_default_scores(self) -> None:
        """Validators without previous validation knowledge should start
        with default score of 0.0 for each UID. The method can also be
        used to reset the scores in case of an internal error"""

        bt.logging.info("Initiating validator with default scores for each UID")
        self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        bt.logging.info(f"Validation weights have been initialized: {self.scores}")
    
    def update_games():
        pass 
        # Logic to pull in from API and updates games db
    def evaluate_miner(self, minerId):
        # Fetch data from predictions table for the specified minerId
        cursor.execute('''
        SELECT predictions.id, predictions.teamGameId, predictions.predictedOutcome, teamGame.outcome
        FROM predictions
        JOIN teamGame ON predictions.teamGameId = teamGame.id
        WHERE predictions.minerId = ?
        ''', (miner_id,))
        
        # Update the predictionCorrect column based on the comparison
        for row in cursor.fetchall():
            prediction_id, team_game_id, predicted_outcome, actual_outcome = row
            prediction_correct = 1 if predicted_outcome == actual_outcome else 0
            cursor.execute('''
            UPDATE predictions
            SET predictionCorrect = ?
            WHERE id = ?
            ''', (prediction_correct, prediction_id))

        # Commit the changes
        conn.commit()
        conn.close()

        
    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
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
            f"Saved the following state to a file: step: {self.step}, scores: {self.scores}, hotkeys: {self.hotkeys}, last_updated_block: {self.last_updated_block}, blacklisted_miner_hotkeys: {self.blacklisted_miner_hotkeys}"
        )

        
    def reset_validator_state(self, state_path):
        """Inits the default validator state. Should be invoked only
        when an exception occurs and the state needs to reset."""

        # Rename current state file in case manual recovery is needed
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
        """Loads the state of the validator from a file."""

        # Load the state of the validator from file.
        state_path = self.base_path + "/state.pt"
        if path.exists(state_path):
            try:
                bt.logging.info("Loading validator state.")
                state = torch.load(state_path)
                bt.logging.debug(f"Loaded the following state from file: {state}")
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.last_updated_block = state["last_updated_block"]
                if "blacklisted_miner_hotkeys" in state.keys():
                    self.blacklisted_miner_hotkeys = state["blacklisted_miner_hotkeys"]

                bt.logging.info(f"Scores loaded from saved file: {self.scores}")
            except Exception as e:
                bt.logging.error(
                    f"Validator state reset because an exception occurred: {e}"
                )
                self.reset_validator_state(state_path=state_path)

        else:
            self.init_default_scores()
            
    def sync_metagraph(self, metagraph, subtensor):
        """Syncs the metagraph"""

        bt.logging.debug(
            f"Syncing metagraph: {self.metagraph} with subtensor: {self.subtensor}"
        )

        # Sync the metagraph
        metagraph.sync(subtensor=subtensor)

        return metagraph


# Need func to set weights; dont think i should take fans?

    def _get_local_miner_blacklist(self) -> list:
        """Returns the blacklisted miners hotkeys from the local file."""

        # Check if local blacklist exists
        blacklist_file = f"{self.base_path}/miner_blacklist.json"
        if Path(blacklist_file).is_file():
            # Load the contents of the local blaclist
            bt.logging.trace(f"Reading local blacklist file: {blacklist_file}")
            try:
                with open(blacklist_file, "r", encoding="utf-8") as file:
                    file_content = file.read()

                miner_blacklist = json.loads(file_content)
                if validate_miner_blacklist(miner_blacklist):
                    bt.logging.trace(f"Loaded miner blacklist: {miner_blacklist}")
                    return miner_blacklist

                bt.logging.trace(
                    f"Loaded miner blacklist was formatted incorrectly or was empty: {miner_blacklist}"
                )
            except OSError as e:
                bt.logging.error(f"Unable to read blacklist file: {e}")
            except json.JSONDecodeError as e:
                bt.logging.error(
                    f"Unable to parse JSON from path: {blacklist_file} with error: {e}"
                )
        else:
            bt.logging.trace(f"No local miner blacklist file in path: {blacklist_file}")

        return []

    def get_uids_to_query(self, all_axons) -> list:
        """Returns the list of UIDs to query"""

        # Get UIDs with a positive stake
        uids_with_stake = self.metagraph.total_stake >= 0.0
        bt.logging.trace(f"UIDs with a positive stake: {uids_with_stake}")

        # Get UIDs with an IP address of 0.0.0.0
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
        bt.logging.trace(f"UIDs with 0.0.0.0 as an IP address: {invalid_uids}")

        # Get UIDs that have their hotkey blacklisted
        blacklisted_uids = []
        if self.blacklisted_miner_hotkeys:
            for hotkey in self.blacklisted_miner_hotkeys:
                if hotkey in self.metagraph.hotkeys:
                    blacklisted_uids.append(self.metagraph.hotkeys.index(hotkey))
                else:
                    bt.logging.trace(
                        f"Blacklisted hotkey {hotkey} was not found from metagraph"
                    )

            bt.logging.debug(f"Blacklisted the following UIDs: {blacklisted_uids}")

        # Convert blacklisted UIDs to tensor
        blacklisted_uids_tensor = torch.tensor(
            [uid not in blacklisted_uids for uid in self.metagraph.uids.tolist()],
            dtype=torch.bool,
        )

        bt.logging.trace(f"Blacklisted UIDs: {blacklisted_uids_tensor}")

        # Determine the UIDs to filter
        uids_to_filter = torch.logical_not(
            ~blacklisted_uids_tensor | ~invalid_uids | ~uids_with_stake
        )

        bt.logging.trace(f"UIDs to filter: {uids_to_filter}")

        # Define UIDs to query
        uids_to_query = [
            axon
            for axon, keep_flag in zip(all_axons, uids_to_filter)
            if keep_flag.item()
        ]

        # Define UIDs to filter
        final_axons_to_filter = [
            axon
            for axon, keep_flag in zip(all_axons, uids_to_filter)
            if not keep_flag.item()
        ]

        uids_not_to_query = [
            self.metagraph.hotkeys.index(axon.hotkey) for axon in final_axons_to_filter
        ]

        bt.logging.trace(f"Final axons to filter: {final_axons_to_filter}")
        bt.logging.debug(f"Filtered UIDs: {uids_not_to_query}")

        # Reduce the number of simultaneous UIDs to query
        if self.max_targets < 256:
            start_idx = self.max_targets * self.target_group
            end_idx = min(
                len(uids_to_query), self.max_targets * (self.target_group + 1)
            )
            if start_idx == end_idx:
                return [], []
            if start_idx >= len(uids_to_query):
                raise IndexError(
                    "Starting index for querying the miners is out-of-bounds"
                )

            if end_idx >= len(uids_to_query):
                end_idx = len(uids_to_query)
                self.target_group = 0
            else:
                self.target_group += 1

            bt.logging.debug(
                f"List indices for UIDs to query starting from: '{start_idx}' ending with: '{end_idx}'"
            )
            uids_to_query = uids_to_query[start_idx:end_idx]

        list_of_uids = [
            self.metagraph.hotkeys.index(axon.hotkey) for axon in uids_to_query
        ]

        list_of_hotkeys = [axon.hotkey for axon in uids_to_query]

        bt.logging.trace(f"Sending query to the following hotkeys: {list_of_hotkeys}")

        return uids_to_query, list_of_uids, blacklisted_uids, uids_not_to_query
    
    def update_game_outcome(self, game_id, outcome):
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE game_data SET outcome = ?, active = 0 WHERE id = ?", (outcome, game_id))
        conn.commit()
        conn.close()

    def get_recent_games(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        three_days_ago = datetime.now() - timedelta(hours=72)
        three_days_ago_str = three_days_ago.isoformat()
        cursor.execute("SELECT id, teamA, teamB, externalId FROM game_data WHERE eventStartDate >= ? AND active = 1", (three_days_ago_str,))
        return cursor.fetchall()

    def determine_winner(self, game_info):
        game_id, teamA, teamB, externalId = game_info

        url = "https://api-baseball.p.rapidapi.com/games"
        headers = {
            "x-rapidapi-host": "api-baseball.p.rapidapi.com",
            "x-rapidapi-key": "b416b1c26dmsh6f20cd13ee1f7ccp11cc1djsnf64975aaacde"
        }
        querystring = {"id": str(externalId)}

        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            data = response.json()
            game_response = data.get('response', [])[0]

            home_team = game_response['teams']['home']['name']
            away_team = game_response['teams']['away']['name']
            home_score = game_response['scores']['home']['total']
            away_score = game_response['scores']['away']['total']

            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    winner = teamA if teamA == home_team else teamB
                elif away_score > home_score:
                    winner = teamB if teamB == away_team else teamA
                else:
                    winner = "Tie"

                self.update_game_outcome(game_id, winner)
                bt.logging.info(f"Game ID: {game_id}, Winner: {winner}")

    def update_recent_games(self):
        recent_games = self.get_recent_games()
        for game_info in recent_games:
            self.determine_winner(game_info)


    def set_weights(self):
        bt.logging.debug(f"Setting weights for validator")
        
