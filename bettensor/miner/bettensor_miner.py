from argparse import ArgumentParser
from typing import Tuple
import sys
import bittensor as bt
import sqlite3
from bettensor.base.neuron import BaseNeuron
from bettensor.protocol import Metadata, GameData, TeamGame, TeamGamePrediction
from bettensor.utils.sign_and_validate import verify_signature
from bettensor.utils.miner_stats import MinerStatsHandler
import datetime
import os
import threading
from contextlib import contextmanager
from bettensor.utils.database_manager import get_db_manager

class BettensorMiner(BaseNeuron):
    """
    The BettensorMiner class contains all of the code for a Miner neuron

    Attributes:
        neuron_config:
            This attribute holds the configuration settings for the neuron:
            bt.subtensor, bt.wallet, bt.logging & bt.axon
        miner_set_weights:
            A boolean attribute that determines whether the miner sets weights.
            This is set based on the command-line argument args.miner_set_weights.
        wallet:
            Represents an instance of bittensor.wallet returned from the setup() method.
        subtensor:
            An instance of bittensor.subtensor returned from the setup() method.
        metagraph:
            An instance of bittensor.metagraph returned from the setup() method.
        miner_uid:
            An int instance representing the unique identifier of the miner in the network returned
            from the setup() method.
        hotkey_blacklisted:
            A boolean flag indicating whether the miner's hotkey is blacklisted.

    """

    default_db_path = "./data/miner.db"

    def __init__(self, parser: ArgumentParser):
        """
        Initializes the Miner class.

        Arguments:
            parser:
                An ArgumentParser instance.

        Returns:
            None
        """
        super().__init__(parser=parser, profile="miner")
        # Allow user to specify db path, if they want to.
        parser.add_argument("--db_path", type=str, default=self.default_db_path)
        # Neuron configuration
        self.neuron_config = self.config(
            bt_classes=[bt.subtensor, bt.logging, bt.wallet, bt.axon]
        )
        args = parser.parse_args()
        # TODO If users want to run a dual miner/vali. Not fully implemented yet.
        # if args.miner_set_weights == "False":
        #     self.miner_set_weights = False
        # else:
        #     self.miner_set_weights = True

        # Minimum stake for validator whitelist
        self.validator_min_stake = args.validator_min_stake

        # Neuron setup
        self.wallet, self.subtensor, self.metagraph, self.miner_uid = self.setup()
        self.hotkey_blacklisted = False
        self.hotkey = self.wallet.hotkey.ss58_address
        
        self.db_path = args.db_path
        os.environ[f'MINER_{self.miner_uid}_DB_PATH'] = self.db_path
        self.db_manager = get_db_manager(self.miner_uid)

        os.environ["DB_PATH"] = self.db_path
        os.environ["HOTKEY"] = self.wallet.hotkey.ss58_address
        os.environ["UID"] = str(self.miner_uid)

        # Initialize local sqlite
        self.ensure_db_directory_exists()
        self.initialize_database()

        # TODO: Refactor to use miner_stats table, no need for txt file
        os.makedirs(os.path.dirname("data/miner_env.txt"), exist_ok=True)

        # Check if the miner's hotkey is already in the file
        if not self.hotkey_exists_in_file("data/miner_env.txt", self.wallet.hotkey.ss58_address):
            with open("data/miner_env.txt", "a") as f:
                f.write(
                    f"UID={self.miner_uid}, DB_PATH={self.db_path}, HOTKEY={self.wallet.hotkey.ss58_address}\n"
                )
            bt.logging.info(f"Added miner info to data/miner_env.txt")
        else:
            bt.logging.info(f"Miner info already exists in data/miner_env.txt")

        # Initialize Miner Stats
        self.stats = MinerStatsHandler(self)
        bt.logging.trace(
            f"Miner stats initialized with miner instance"
        )
        init_stats = self.stats.init_miner_row(
            self.wallet.hotkey.ss58_address, self.miner_uid
        )
        if not init_stats:
            bt.logging.error(
                f"Failed to initialize miner stats. Submitting predictions will not work properly"
            )
        bt.logging.debug(f"init_stats: {init_stats}")
        self.stats.reset_daily_cash_on_startup()

    def ensure_db_directory_exists(self):
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def print_table_schema(self):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(games)")
            schema = cursor.fetchall()
            for column in schema:
                print(column)

    def initialize_database(self):
        bt.logging.debug(f"Initializing database at {self.db_path}")
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
                cursor.connection.commit()
        except sqlite3.Error as e:
            bt.logging.error(f"Failed to initialize local database: {e}")
            raise Exception("Failed to initialize local database")

    def setup(self) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph, str]:
        """

        The setup function initializes the neuron by registering the
        configuration.

        Arguments:
            None

        Returns:
            wallet:
                An instance of bittensor.wallet containing information about
                the wallet
            subtensor:
                An instance of bittensor.subtensor
            metagraph:
                An instance of bittensor.metagraph
            miner_uid:
                An instance of int consisting of the miner UID

        Raises:
            AttributeError:
                The AttributeError is raised if wallet, subtensor & metagraph cannot be logged.
        """
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        bt.logging.info(
            f"Initializing miner for subnet: {self.neuron_config.netuid} on network: {self.neuron_config.subtensor.chain_endpoint} with config:\n {self.neuron_config}"
        )

        # Setup the bittensor objects
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

        # Validate that our hotkey can be found from metagraph
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"Your miner: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again"
            )
            sys.exit()

        # Get the unique identity (UID) from the network
        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Miner is running with UID: {miner_uid}")

        return wallet, subtensor, metagraph, miner_uid

    def check_whitelist(self, hotkey):
        """
        Checks if a given validator hotkey has been whitelisted.

        Arguments:
            hotkey:
                A str instance depicting a hotkey.

        Returns:
            True:
                True is returned if the hotkey is whitelisted.
            False:
                False is returned if the hotkey is not whitelisted.
        """

        if isinstance(hotkey, bool) or not isinstance(hotkey, str):
            return False

        whitelisted_hotkeys = []

        if hotkey in whitelisted_hotkeys:
            return True

        return False

    def blacklist(self, synapse: GameData) -> Tuple[bool, str]:
        """
        This function is executed before the synapse data has been
        deserialized.

        On a practical level this means that whatever blacklisting
        operations we want to perform, it must be done based on the
        request headers or other data that can be retrieved outside of
        the request data.

        As it currently stands, we want to blacklist requests that are
        not originating from valid validators. This includes:
        - unregistered hotkeys
        - entities which are not validators
        - entities with insufficient stake

        Returns:
            [True, ""] for blacklisted requests where the reason for
            blacklisting is contained in the quotes.
            [False, ""] for non-blacklisted requests, where the quotes
            contain a formatted string (f"Hotkey {synapse.dendrite.hotkey}
            has insufficient stake: {stake}",)
        """

        # Check whitelisted hotkeys (queries should always be allowed)
        if self.check_whitelist(hotkey=synapse.dendrite.hotkey):
            bt.logging.info(f"Accepted whitelisted hotkey: {synapse.dendrite.hotkey})")
            return (False, f"Accepted whitelisted hotkey: {synapse.dendrite.hotkey}")

        # Blacklist entities that have not registered their hotkey
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.info(f"Blacklisted unknown hotkey: {synapse.dendrite.hotkey}")
            return (
                True,
                f"Hotkey {synapse.dendrite.hotkey} was not found from metagraph.hotkeys",
            )

        # Blacklist entities that are not validators
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        print("uid:", uid)
        print("metagraph", self.metagraph)
        if not self.metagraph.validator_permit[uid]:
            bt.logging.info(f"Blacklisted non-validator: {synapse.dendrite.hotkey}")
            return (True, f"Hotkey {synapse.dendrite.hotkey} is not a validator")


        bt.logging.info(f"validator_min_stake: {self.validator_min_stake}")
        # Blacklist entities that have insufficient stake
        stake = float(self.metagraph.S[uid])
        if stake < self.validator_min_stake:
            bt.logging.info(
                f"Blacklisted validator {synapse.dendrite.hotkey} with insufficient stake: {stake}"
            )
            return (
                True,
                f"Hotkey {synapse.dendrite.hotkey} has insufficient stake: {stake}",
            )

        # Allow all other entities
        bt.logging.info(
            f"Accepted hotkey: {synapse.dendrite.hotkey} (UID: {uid} - Stake: {stake})"
        )
        return (False, f"Accepted hotkey: {synapse.dendrite.hotkey}")

    def priority(self, synapse: GameData) -> float:
        """
        Assigns a priority to the synapse based on the stake of the validator.
        """

        # Prioritize whitelisted validators
        if self.check_whitelist(hotkey=synapse.dendrite.hotkey):
            return 10000000.0

        # Otherwise prioritize validators based on their stake
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = float(self.metagraph.S[uid])

        print(f"Prioritized: {synapse.dendrite.hotkey} (UID: {uid} - Stake: {stake})")

        return stake

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
                f"Received a synapse from a validator with lower subnet version ({synapse.metadata.subnet_version}) than yours ({self.subnet_version})."
            )

        # TODO: METADATA / Signature Verification

        game_data_dict = synapse.gamedata_dict
        self.add_game_data(game_data_dict)

        # check if tables in db are initialized
        # if not, initialize them
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games")
            if not cursor.fetchone():
                self.initialize_database()

        # clean up games table and set active field
        self.update_games_data()

        # Remove duplicate games
        self.remove_duplicate_games()

        # Get current time
        current_time = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="minutes"
        )
        # Update outcomes for all predictions and recalculate miner stats
        

        # Fetch games that have not started yet
        with self.db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT externalID FROM games WHERE eventStartDate > ?", (current_time,)
            )
            games = cursor.fetchall()

        bt.logging.debug(f"Fetched {len(games)} games")

        # Process the fetched games
        bt.logging.info(f"Processing recent predictions")
        prediction_dict = {}
        for game in games:
            bt.logging.trace(f" Processing Predictions: Game: {game}")
            external_game_id = game[0]
            bt.logging.trace(f"Processing Predictions: Game ID: {external_game_id}")

            # Fetch predictions for the game from the last 3 days
            three_days_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).isoformat(timespec="minutes")
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM predictions WHERE teamGameID = ? AND predictionDate > ?", 
                    (external_game_id, three_days_ago)
                )
                predictions = cursor.fetchall()
            bt.logging.trace(f"Predictions: {predictions}")

            # Add predictions to prediction_dict
            for prediction in predictions:
                if len(prediction) >= 13:
                    single_prediction = TeamGamePrediction(
                        predictionID=prediction[0],
                        teamGameID=prediction[1],
                        minerID=str(self.miner_uid),
                        predictionDate=prediction[3],
                        predictedOutcome=prediction[4],
                        teamA=prediction[5],
                        teamB=prediction[6],
                        wager=prediction[7],
                        teamAodds=prediction[8],
                        teamBodds=prediction[9],
                        tieOdds=prediction[10],
                        can_overwrite=prediction[11],
                        outcome=prediction[12],
                    )
                    prediction_dict[prediction[0]] = single_prediction
                else:
                    bt.logging.warning(f"Skipping prediction due to insufficient data: {prediction}")

        bt.logging.trace(f"prediction_dict: {prediction_dict}")
        synapse.prediction_dict = prediction_dict
        synapse.gamedata_dict = None
        synapse.metadata = Metadata.create(
            wallet=self.wallet,
            subnet_version=self.subnet_version,
            neuron_uid=self.miner_uid,
            synapse_type="prediction",
        )
        self.update_outcomes()
        return synapse

    def add_game_data(self, game_data_dict):
        try:
            bt.logging.trace(f"add_game_data() | Adding game data to local database")
            number_of_games = len(game_data_dict)
            bt.logging.trace(
                f"add_game_data() | Number of games to add: {number_of_games}"
            )
            with self.db_manager.get_cursor() as cursor:
                for game_id, game_data in game_data_dict.items():
                    external_id = game_data.externalId
                    cursor.execute("SELECT * FROM games WHERE externalID = ?", (external_id,))
                    if not cursor.fetchone():
                        cursor.execute(
                            """INSERT INTO games (
                            gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID, createDate, lastUpdateDate, 
                            eventStartDate, active, outcome, tieOdds, canTie
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                game_id,
                                game_data.teamA,
                                game_data.teamAodds,
                                game_data.teamB,
                                game_data.teamBodds,
                                game_data.sport,
                                game_data.league,
                                game_data.externalId,
                                game_data.createDate,
                                game_data.lastUpdateDate,
                                game_data.eventStartDate,
                                game_data.active,
                                game_data.outcome,
                                game_data.tieOdds,
                                game_data.canTie,
                            ),
                        )
                    else:
                        bt.logging.trace(
                            f"add_game_data() | Game {external_id} already in the table, updating."
                        )
                        cursor.execute(
                            """UPDATE games SET teamA = ?, teamAodds = ?, teamB = ?, teamBodds = ?, sport = ?, league = ?, externalID = ?, 
                                       createDate = ?, lastUpdateDate = ?, eventStartDate = ?, active = ?, outcome = ?, tieOdds = ?, canTie = ? WHERE gameID = ?""",
                            (
                                game_data.teamA,
                                game_data.teamAodds,
                                game_data.teamB,
                                game_data.teamBodds,
                                game_data.sport,
                                game_data.league,
                                game_data.externalId,
                                game_data.createDate,
                                game_data.lastUpdateDate,
                                game_data.eventStartDate,
                                game_data.active,
                                game_data.outcome,
                                game_data.tieOdds,
                                game_data.canTie,
                                game_id,
                            ),
                        )
                cursor.connection.commit()
        except Exception as e:
            bt.logging.error(f"Failed to add game data: {e}")

    def update_games_data(self):
        bt.logging.trace(f"update_games_data() | Updating games data")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games")
            games = cursor.fetchall()
            for game in games:
                bt.logging.trace(
                    f"Current time: {datetime.datetime.now(datetime.timezone.utc)}"
                )
                bt.logging.trace(f"update_games_data() | Game: {game}")

                if game[10] is None:
                    bt.logging.trace("update_games_data() | Start Date is None, passing")
                    continue
                
                event_start_date = datetime.datetime.fromisoformat(game[10].replace('Z', '+00:00'))

                current_time = datetime.datetime.now(datetime.timezone.utc)
                if current_time > event_start_date:
                    cursor.execute(
                        "UPDATE games SET active = 1 WHERE gameID = ?", (game[0],)
                    )
                    bt.logging.trace(f"update_games_data() | Game {game[0]} is now active")

                if current_time > event_start_date + datetime.timedelta(days=3):
                    cursor.execute("DELETE FROM games WHERE gameID = ?", (game[0],))
                    bt.logging.trace(
                        f"update_games_data() | Game {game[0]} is deleted from db"
                    )

                if current_time < event_start_date:
                    cursor.execute(
                        "UPDATE games SET active = 0 WHERE gameID = ?", (game[0],)
                    )
                    bt.logging.trace(
                        f"update_games_data() | Game {game[0]} is now inactive"
                    )

            cursor.connection.commit()

    def hotkey_exists_in_file(self, file_path, hotkey):
        if not os.path.exists(file_path):
            return False
        with open(file_path, "r") as f:
            for line in f:
                if f"HOTKEY={hotkey}" in line:
                    return True
        return False

    def remove_duplicate_games(self):
        bt.logging.trace("Removing duplicate games and predictions from the database")
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT externalID, COUNT(*) as count
                    FROM games
                    GROUP BY externalID
                    HAVING count > 1
                """)
                duplicates = cursor.fetchall()

                for external_id, count in duplicates:
                    bt.logging.debug(f"Found {count} duplicates for externalID: {external_id}")
                    
                    cursor.execute("""
                        DELETE FROM games
                        WHERE externalID = ? AND rowid NOT IN (
                            SELECT rowid
                            FROM games
                            WHERE externalID = ?
                            ORDER BY lastUpdateDate DESC
                            LIMIT 1
                        )
                    """, (external_id, external_id))

                cursor.execute("""
                    SELECT teamGameID, COUNT(*) as count
                    FROM predictions
                    GROUP BY teamGameID
                    HAVING count > 1
                """)

                prediction_duplicates = cursor.fetchall()
                for team_game_id, count in prediction_duplicates:
                    bt.logging.debug(f"Found {count} duplicates for teamGameID: {team_game_id}")
                
                    cursor.execute("""
                        DELETE FROM predictions
                        WHERE teamGameID = ? AND rowid NOT IN (
                            SELECT rowid
                            FROM predictions
                            WHERE teamGameID = ?
                            ORDER BY predictionDate DESC
                            LIMIT 1
                        )
                    """, (team_game_id, team_game_id))

                cursor.connection.commit()
            bt.logging.trace(f"Removed {len(duplicates)} sets of duplicate games")
            bt.logging.trace(f"Removed {len(prediction_duplicates)} sets of duplicate predictions")
        except sqlite3.Error as e:
            bt.logging.error(f"Error removing duplicate games or predictions: {e}")

    def get_predictions(self):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM predictions")
            predictions_raw = cursor.fetchall()

        prediction_dict = {}
        for prediction in predictions_raw:
            bt.logging.trace(f"get_predictions() | Prediction: {prediction}")
            single_prediction = TeamGamePrediction(
                predictionID=prediction[0],
                teamGameID=prediction[1],
                minerID=prediction[2] or self.miner_uid,
                predictionDate=prediction[3],
                predictedOutcome=prediction[4],
                teamA=prediction[5],
                teamB=prediction[6],
                wager=prediction[7],
                teamAodds=prediction[8],
                teamBodds=prediction[9],
                tieOdds=prediction[10],
                can_overwrite=prediction[11],
                outcome=prediction[12],
            )   
            prediction_dict[prediction[0]] = single_prediction
        prediction_dict = dict(sorted(prediction_dict.items(), key=lambda item: item[1].predictionDate, reverse=True))

        return prediction_dict

    def get_games(self):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games")
            games_raw = cursor.fetchall()

        game_dict = {}
        for game in games_raw:
            single_game = TeamGame(
                id=game[0],
                teamA=game[1],
                teamAodds=game[2],
                teamB=game[3],
                teamBodds=game[4],
                sport=game[5],
                league=game[6],
                externalId=game[7],
                createDate=game[8],
                lastUpdateDate=game[9],
                eventStartDate=game[10],
                active=game[11],
                outcome=game[12],
                tieOdds=game[13],
                canTie=game[14],
            )
            game_dict[game[0]] = single_game
        return game_dict

    def update_outcomes(self):
        bt.logging.info("update_outcomes() | Updating outcomes for all predictions and recalculating miner stats")
        prediction_dict = self.get_predictions()
        game_dict = self.get_games()

        current_time = datetime.datetime.now(datetime.timezone.utc)
        current_stats = self.stats.return_miner_stats(self.hotkey)
        
        for prediction_id, prediction in prediction_dict.items():
            if prediction.teamGameID in game_dict:
                game = game_dict[prediction.teamGameID]
                game_start_time = datetime.datetime.fromisoformat(game.eventStartDate.replace('Z', '+00:00'))
                
                # Only process games that have already started
                if current_time > game_start_time:
                    outcome = game.outcome
                    if outcome == "Unfinished":
                        continue
                
                    if outcome == 0:  # teamA wins
                        if prediction.predictedOutcome == prediction.teamA:
                            self.update_prediction_outcome(prediction, "Win", current_stats, prediction.teamAodds)
                        else:
                            self.update_prediction_outcome(prediction, "Loss", current_stats)
                    elif outcome == 1:  # teamB wins
                        if prediction.predictedOutcome == prediction.teamB:
                            self.update_prediction_outcome(prediction, "Win", current_stats, prediction.teamBodds)
                        else:
                            self.update_prediction_outcome(prediction, "Loss", current_stats)
                    elif outcome == 2:  # tie
                        if prediction.predictedOutcome == "Tie":
                            self.update_prediction_outcome(prediction, "Win", current_stats, prediction.tieOdds)
                        else:
                            self.update_prediction_outcome(prediction, "Loss", current_stats)

        # Recalculate ratio
        total_games = current_stats.miner_lifetime_wins + current_stats.miner_lifetime_losses
        
        if total_games == 0:
            current_stats.miner_win_loss_ratio = 0  # No games played
        else:
            current_stats.miner_win_loss_ratio = current_stats.miner_lifetime_wins / total_games

        # Round to 3 decimal places for precision
        current_stats.miner_win_loss_ratio = round(current_stats.miner_win_loss_ratio, 3)

        # Get most recent prediction date from prediction dict
        if prediction_dict:
            current_stats.miner_last_prediction_date = max(
                prediction.predictionDate for prediction in prediction_dict.values()
            )
        else:
            current_stats.miner_last_prediction_date = None

        self.stats.update_miner_row(current_stats)
    
    def update_prediction_outcome(self, prediction, outcome, current_stats, odds=None):
        if prediction.outcome != outcome:  # Only update if the outcome has changed
            prediction.outcome = outcome
            if outcome == "Win":
                current_stats.miner_lifetime_wins += 1
                if odds:
                    current_stats.miner_lifetime_earnings += prediction.wager * odds
            else:  # Loss
                current_stats.miner_lifetime_losses += 1
            
            # Update the prediction in the database
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(
                    "UPDATE predictions SET outcome = ? WHERE predictionID = ?",
                    (outcome, prediction.predictionID)
                )
                cursor.connection.commit()
        
