from argparse import ArgumentParser
from typing import Tuple
import sys
from uuid import UUID
import requests
import bittensor as bt
import sqlite3
from bettensor.base.neuron import BaseNeuron
from bettensor.protocol import Metadata, GameData, Prediction, TeamGamePrediction
from bettensor.miner import cli
from bettensor import validate_miner_blacklist, validate_signature
import datetime
import os

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

        self.neuron_config = self.config(
            bt_classes=[bt.subtensor, bt.logging, bt.wallet, bt.axon]
        )

        args = parser.parse_args()
        if args.miner_set_weights == "False":
            self.miner_set_weights = False
        else:
            self.miner_set_weights = True

        self.validator_min_stake = args.validator_min_stake

        self.wallet, self.subtensor, self.metagraph, self.miner_uid = self.setup()

        self.hotkey_blacklisted = False

        # Initialize local sqlite
        self.db_path = './miner.db'
        self.ensure_db_directory_exists(self.db_path)
        self.initialize_database()

    def ensure_db_directory_exists(self, db_path):
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def initialize_database(self):
        try:
            db = sqlite3.connect(self.db_path)
            cursor = db.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                               predictionID TEXT PRIMARY KEY, 
                               teamGameID TEXT, 
                               minerID UUID, 
                               predictionDate TEXT, 
                               predictedOutcome TEXT, 
                               wager REAL,
                               teamAodds REAL,
                               teamBodds REAL,
                               tieOdds REAL,
                               outcome TEXT
                               )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS games (
                               gameID TEXT PRIMARY KEY, 
                               teamA TEXT,
                               teamAOdds REAL,
                               teamB TEXT,
                               teamBodds REAL,
                               sport TEXT, 
                               league TEXT, 
                               eventDescription TEXT, 
                               externalID TEXT, 
                               createDate TEXT, 
                               lastUpdateDate TEXT, 
                               eventStartDate TEXT, 
                               active BOOLEAN, 
                               outcome TEXT
                               )''')
            db.commit()
            db.close()
        except sqlite3.Error as e:
            bt.logging.error(f"Failed to initialize local database: {e}")
            raise Exception("Failed to initialize local database")
    @classmethod
    def get_cursor(cls):
        try:
            db = sqlite3.connect(cls.db_path)
            return db, db.cursor()
        except sqlite3.Error as e:
            bt.logging.error(f"Failed to connect to local database: {e}")
            raise Exception("Failed to connect to local database")

    def setup(self) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph, str]:
        """This function sets up the neuron.

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

        whitelisted_hotkeys = [
        ]

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

        print(
            f"Prioritized: {synapse.dendrite.hotkey} (UID: {uid} - Stake: {stake})"
        )

        return stake

    def forward(self, synapse: GameData) -> Prediction:
        """
        
        """
        db, cursor = self.get_cursor()

        # Print version information and perform version checks
        print(
            f"Synapse version: {synapse.subnet_version}, our version: {self.subnet_version}"
        )
        if synapse.subnet_version > self.subnet_version:
            bt.logging.warning(
                f"Received a synapse from a validator with higher subnet version ({synapse.subnet_version}) than yours ({self.subnet_version}). Please update the miner, or you may encounter issues."
            )

        # Synapse signature verification
        data = f'{synapse.synapse_nonce}{synapse.synapse_timestamp}'
        if not validate_signature(
            hotkey=synapse.dendrite.hotkey,
            data=data,
            signature=synapse.synapse_signature,
        ):
            print(
                f"Failed to validate signature for the synapse. Hotkey: {synapse.dendrite.hotkey}, data: {data}, signature: {synapse.synapse_signature}"
            )
            return synapse
        else:
            print(
                f"Succesfully validated signature for the synapse. Hotkey: {synapse.dendrite.hotkey}, data: {data}, signature: {synapse.synapse_signature}"
            )


        
        synapse_timestamp = synapse.metadata.timestamp
        synapse_id = synapse.metadata.id
        validator_id = synapse.metadata.neuron_id
        server_subnet_version = synapse.metadata.subnet_version
        bt.logging.info(f"Received synapse from validator: {validator_id} with ID: {synapse_id} subnet version: {server_subnet_version} and timestamp: {synapse_timestamp}")
        
        #update games table
        games_dict = synapse.gamedata_dict
        bt.logging.info(f"Received {len(games_dict)} games, updating games table in local database")
        for game in games_dict:
            #if UUID.to_string not in games table, insert
            if game not in cursor.execute('SELECT * FROM games WHERE gameId = ?', (game.to_string(),)):
                cursor.execute('''INSERT INTO games (
                               gameID, 
                               teamA, 
                               teamAOdds, 
                               teamB, 
                               teamBodds, 
                               sport, 
                               league, 
                               eventDescription, 
                               externalId, 
                               createDate, 
                               lastUpdateDate, 
                               eventStartDate, 
                               active, 
                               outcome) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                               (game.to_string(), 
                                game.teamA, 
                                game.teamAOdds, 
                                game.teamB, 
                                game.teamBodds, 
                                game.sport, 
                                game.league, 
                                game.eventDescription, 
                                game.externalId, 
                                game.createDate, 
                                game.lastUpdateDate, 
                                game.eventStartDate, 
                                str(game.active), #bool to string
                                game.outcome))
                db.commit()
            else:
                # if game.id in games table, update to latest data
                cursor.execute('''
                               UPDATE games SET teamA = ?, 
                               teamAOdds = ?, 
                               teamB = ?, 
                               teamBodds = ?, 
                               sport = ?, 
                               league = ?, 
                               eventDescription = ?, 
                               externalId = ?, 
                               createDate = ?, 
                               lastUpdateDate = ?, 
                               eventStartDate = ?, 
                               active = ?, 
                               outcome = ? 
                               WHERE gameId = ?''', 
                               (game.teamA, 
                                game.teamAOdds, 
                                game.teamB, 
                                game.teamBodds, 
                                game.sport, 
                                game.league, 
                                game.eventDescription, 
                                game.externalId, 
                                game.createDate, 
                                game.lastUpdateDate, 
                                game.eventStartDate, 
                                str(game.active), 
                                game.outcome, 
                                game.id))
                db.commit()
        # construct prediction 
        prediction_dict = {}

        #get games that have not started yet
        games = cursor.execute('SELECT gameID FROM games WHERE eventStartDate < ?', (datetime.now().isoformat()))

        #get predictions for these games
        predictions = cursor.execute('SELECT * FROM predictions WHERE gameID IN (?)', (games))

        # add predictions to prediction_dict
        for prediction in predictions:
            single_prediction = TeamGamePrediction(
                predictionID = UUID(prediction[0]),
                teamGameID = UUID(prediction[1]),
                minerID = UUID(prediction[2]),
                predictionDate = prediction[3],
                predictedOutcome = prediction[4],
                wager = prediction[5],
                teamAodds = prediction[6],
                teamBodds = prediction[7],
                tieOdds = prediction[8],
                outcome = prediction[9]
            )
            prediction_dict[UUID(prediction[0])] = single_prediction
            
        try:
            metadata = Metadata.create(self.wallet, self.subnet_version, self.miner_uid)
        except Exception as e:
            bt.logging.error(f"Failed to create metadata: {e}")
            raise Exception("Failed to create metadata")
        try:
            prediction_synapse = Prediction.create(metadata, prediction_dict)
        except Exception as e:
            bt.logging.error(f"Failed to create prediction synapse: {e}")
            raise Exception("Failed to create prediction synapse")

        return prediction_synapse

