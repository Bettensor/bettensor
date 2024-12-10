"""
Class to handle and process all incoming miner data.
"""


import asyncio
import datetime
from datetime import datetime, timezone, timedelta
import traceback
from typing import Dict
import bittensor as bt
from pydantic import ValidationError
import torch
from bettensor.protocol import GameData, TeamGame, TeamGamePrediction
from bettensor.validator.utils.database.database_manager import DatabaseManager


"""
Miner Data Methods, Extends the Bettensor Validator Class

"""


class MinerDataMixin:
    async def insert_predictions(self, processed_uids, predictions):
        """
        Inserts new predictions into the database in batches
        """
        self.processed_uids = processed_uids
        
        # Initialize validation stats for this batch
        validation_stats = {
            'duplicate_prediction': 0,
            'uid_not_processed': 0,
            'invalid_uid': 0,
            'invalid_wager': 0,
            'daily_limit_exceeded': 0,
            'game_not_found': 0,
            'game_started': 0,
            'invalid_outcome': 0,
            'invalid_odds': 0,
            'invalid_confidence': 0,
            'other_errors': 0,
            'successful': 0
        }

        try:
            current_time = datetime.now(timezone.utc).isoformat()
            bt.logging.trace(f"insert_predictions called with {len(predictions)} predictions")

            return_dict = {}
            valid_predictions = []

            # Process predictions first without database writes
            for miner_uid, prediction_dict in predictions.items():
                for prediction_id, res in prediction_dict.items():
                    # Validate the prediction
                    is_valid, message, validation_type = await self.validate_prediction(
                        miner_uid, 
                        prediction_id, 
                        res.dict()  # Convert Pydantic model to dict
                    )
                    
                    # Update validation stats
                    if is_valid:
                        validation_stats['successful'] += 1
                    else:
                        validation_stats[validation_type] += 1
                    
                    return_dict[prediction_id] = (is_valid, message)
                    
                    if is_valid:
                        # Convert predicted outcome to numeric
                        if res.predicted_outcome == res.team_a:
                            predicted_outcome = 0
                        elif res.predicted_outcome == res.team_b:
                            predicted_outcome = 1
                        else:
                            predicted_outcome = 2
                        res.outcome = 3

                        # Add to valid predictions for batch insert
                        valid_predictions.append((
                            prediction_id, res.game_id, miner_uid, current_time,
                            predicted_outcome, res.predicted_odds, res.team_a, res.team_b,
                            res.wager, res.team_a_odds, res.team_b_odds, res.tie_odds,
                            res.outcome, res.model_name, res.confidence_score
                        ))
            bt.logging.info(f"Validated {len(valid_predictions)} predictions")

            # Log validation statistics for this batch
            bt.logging.info("Batch Validation Statistics:")
            total_predictions = sum(validation_stats.values())
            for reason, count in validation_stats.items():
                if count > 0:
                    percentage = (count / total_predictions) * 100
                    bt.logging.info(f"  {reason}: {count} ({percentage:.1f}%)")

            # Handle database operations for valid predictions
            if valid_predictions:
                bt.logging.info(f"Inserting {len(valid_predictions)} predictions into database")
                try:
                    
                    try:
                        await self.db_manager.executemany(
                            """
                            INSERT INTO predictions (
                                prediction_id, game_id, miner_uid, prediction_date,
                                predicted_outcome, predicted_odds, team_a, team_b,
                                wager, team_a_odds, team_b_odds, tie_odds,
                                outcome, model_name, confidence_score
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            valid_predictions
                        )
                        
                        # Update entropy system after successful commit
                        for pred_values in valid_predictions:
                            try:
                                self.scoring_system.entropy_system.add_prediction(
                                    pred_values[0], pred_values[2], pred_values[1],
                                    pred_values[4], pred_values[8], pred_values[5],
                                    pred_values[3]
                                )
                                return_dict[pred_values[0]] = (True, "Prediction inserted successfully")
                            except Exception as e:
                                bt.logging.error(f"Error updating entropy system: {str(e)}")
                                return_dict[pred_values[0]] = (False, f"Error updating entropy: {str(e)}")
                        
                        
                    except Exception as e:
                        
                        raise e
                       
                except Exception as e:
                    bt.logging.error(f"Error in batch processing: {e}")
                    # Mark all predictions in this batch as failed
                    for pred_values in valid_predictions:
                        prediction_id = pred_values[0]
                        return_dict[prediction_id] = (False, f"Database error: {str(e)}")

            # After successful database operations, send confirmations
            BATCH_SIZE = 16
            MAX_CONCURRENT_SENDS = 4

            # Group miners into batches
            miner_batches = []
            current_batch = []
            
            for miner_uid, prediction_dict in predictions.items():
                current_batch.append((miner_uid, {
                    pred_id: return_dict.get(pred_id, (False, "Unknown prediction ID"))
                    for pred_id in prediction_dict.keys()
                }))
                
                if len(current_batch) >= BATCH_SIZE:
                    miner_batches.append(current_batch)
                    current_batch = []
            
            if current_batch:  # Add any remaining miners
                miner_batches.append(current_batch)

            # Process batches without using async context managers
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_SENDS)

            async def process_confirmation_batch(batch):
                async def send_with_semaphore(miner_uid, miner_predictions):
                    async with semaphore:
                        return await self.send_confirmation_synapse(miner_uid, miner_predictions)

                tasks = [
                    asyncio.create_task(send_with_semaphore(miner_uid, miner_predictions))
                    for miner_uid, miner_predictions in batch
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.gather(*[
                process_confirmation_batch(batch) for batch in miner_batches
            ], return_exceptions=True)

            return return_dict
        
        except Exception as e:
            bt.logging.error(f"Error in insert_predictions: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {k: (False, f"Fatal error: {str(e)}") for k in predictions.keys()}



    async def send_confirmation_synapse(self, miner_uid, predictions):
        """
        Asynchronously sends a confirmation synapse to the miner.

        Args:
            miner_uid: the uid of the miner
            predictions: a dictionary with uids as keys and TeamGamePrediction objects as values
        """
        confirmation_dict = {
            str(pred_id): {"success": success, "message": message}
            for pred_id, (success, message) in predictions.items()
        }

        # Get miner stats for uid asynchronously
        miner_stats = await self.db_manager.fetch_one(
            "SELECT * FROM miner_stats WHERE miner_uid = ?", (miner_uid,)
        )

        if miner_stats is None:
            bt.logging.warning(f"No miner_stats found for miner_uid: {miner_uid}")
            confirmation_dict['miner_stats'] = {}
        else:
            # Handle None values in miner_stats
            for key, value in miner_stats.items():
                if value is None:
                    miner_stats[key] = 0
            confirmation_dict['miner_stats'] = miner_stats

        synapse = GameData.create(
            db_path=self.db_path,
            wallet=self.wallet,
            subnet_version=self.subnet_version,
            neuron_uid=miner_uid,
            synapse_type="confirmation",
            confirmation_dict=confirmation_dict,
        )

        # Convert miner_uid to integer for indexing
        miner_uid_int = int(miner_uid)
        axon = self.metagraph.axons[miner_uid_int]

        bt.logging.info(f"Sending confirmation synapse to miner {miner_uid}, axon: {axon}")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self.dendrite.query,
                axon,
                synapse,
                self.timeout,
                True,
            )
            bt.logging.info(f"Confirmation synapse sent to miner {miner_uid}")
        except Exception as e:
            bt.logging.error(f"An error occurred while sending confirmation synapse: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def process_prediction(self, processed_uids: torch.tensor, synapses: list) -> list:
        """
        processes responses received by miners

        Args:
            processed_uids: list of uids that have been processed
            synapses: list of deserialized synapses
        """
        bt.logging.trace(f"enter process_prediction")
        predictions = {}
        try:
            for synapse in synapses:
                #bt.logging.trace(f"Processing synapse type: {type(synapse)}")
                
                if not hasattr(synapse, 'prediction_dict') or not hasattr(synapse, 'metadata'):
                    bt.logging.warning(f"Invalid synapse object: {synapse}")
                    continue
                
                prediction_dict = synapse.prediction_dict
                metadata = synapse.metadata
                headers = synapse.to_headers()
                #bt.logging.info(f"Synapse headers: {headers}")

                if metadata and hasattr(metadata, "neuron_uid"):
                    uid = metadata.neuron_uid
                    synapse_hotkey = headers.get("bt_header_axon_hotkey")

                    # check synapse hotkey matches hotkey in metagraph
                    if synapse_hotkey != self.metagraph.hotkeys[int(uid)]:
                        bt.logging.warning(f"Synapse hotkey {synapse_hotkey} does not match hotkey {self.metagraph.hotkeys[int(uid)]} for miner {uid}")
                        continue

                    #check that all predictions are submitted with the same uid as the synapse
                    if any(pred.miner_uid != uid for pred in prediction_dict.values()):
                        bt.logging.warning(f"Predictions submitted with different miner UID than synapse: {uid}")
                        continue

                    # for each prediction, check the miner_uid matches the synapse uid and print the number of predictions that match
                    matching_predictions = sum(1 for pred in prediction_dict.values() if pred.miner_uid == uid)
                    #bt.logging.info(f"Number of predictions matching synapse uid: {matching_predictions}")

                    # if any predictions don't match the synapse uid, log a warning, and skip the synapse
                    if matching_predictions != len(prediction_dict):
                        bt.logging.warning(f"Some predictions do not match synapse uid: {uid}")
                        bt.logging.warning(f"Offending uid and hotkey: {uid} {synapse_hotkey}")
                        continue

                    # Ensure prediction_dict is not None before processing
                    if prediction_dict is not None:
                        predictions[uid] = prediction_dict
                    else:
                        bt.logging.trace(
                        f"prediction from miner {uid} is empty and will be skipped."
                    )
                    #bt.logging.info(f"Predictions dict length for uid {uid}: {len(prediction_dict)}")
                else:
                    bt.logging.warning(
                        "metadata is missing or does not contain neuron_uid."
                    )
                
            await self.insert_predictions(processed_uids, predictions)

        except Exception as e:
            bt.logging.error(f"miner_data.py | process_prediction | An error occurred: {e}")
            bt.logging.error(f"miner_data.py | process_prediction | Traceback: {traceback.format_exc()}")
            raise

    def update_recent_games(self):
        bt.logging.info("miner_data.py update_recent_games called")
        current_time = datetime.now(timezone.utc)
        five_hours_ago = current_time - timedelta(hours=4)

        recent_games = self.db_manager.fetch_all(
            """
            SELECT external_id, team_a, team_b, sport, league, event_start_date
            FROM game_data
            WHERE event_start_date < ? AND (outcome = 'Unfinished' OR outcome = 3)
            """,
            (five_hours_ago.isoformat(),),
        )
        bt.logging.info("Recent games: ")
        bt.logging.info(recent_games)

        for game in recent_games:
            external_id, team_a, team_b, sport, league, event_start_date = game
            game_info = {
                "external_id": external_id,
                "team_a": team_a,
                "team_b": team_b,
                "sport": sport,
                "league": league,
                "event_start_date": event_start_date,
            }
            bt.logging.info("Game info: ")
            bt.logging.info(game_info)
            numeric_outcome = self.api_client.determine_winner(game_info)
            bt.logging.info("Outcome: ")
            bt.logging.info(numeric_outcome)

            if numeric_outcome is not None:
                # Update the game outcome in the database
                self.api_client.update_game_outcome(external_id, numeric_outcome)

       
        bt.logging.info(f"Checked {len(recent_games)} games for updates")

    def prepare_game_data_for_entropy(self, predictions):
        game_data = []
        for game_id, game_predictions in predictions.items():
            current_odds = self.get_current_odds(game_id)
            game_data.append(
                {
                    "id": game_id,
                    "predictions": game_predictions,
                    "current_odds": current_odds,
                }
            )
        return game_data

    def get_recent_games(self):
        """retrieves recent games from the database"""
        two_days_ago = (
            datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            - datetime.timedelta(hours=48)
        ).isoformat()
        return self.db_manager.fetch_all(
            "SELECT id, team_a, team_b, external_id FROM game_data WHERE event_start_date >= ? AND outcome = 'Unfinished'",
            (two_days_ago,),
        )

    def get_current_odds(self, game_id):
        try:
            # Query to fetch the current odds for the given game_id
            query = """
            SELECT team_a_odds, team_b_odds, tie_odds
            FROM game_data
            WHERE id = ? OR external_id = ?
            """
            result = self.db_manager.fetchone(query, (game_id, game_id))
            if result:
                home_odds, away_odds, tie_odds = result
                return [home_odds, away_odds, tie_odds]
            else:
                bt.logging.warning(f"No odds found for game_id: {game_id}")
                return [0.0, 0.0, 0.0]  # Return default values if no odds are found
        except Exception as e:
            bt.logging.error(f"Database error in get_current_odds: {e}")
            return [0.0, 0.0, 0.0]  # Return default values in case of database error

    async def fetch_local_game_data(self, current_time):
        """Fetch game data from local database."""
        try:
            # Ensure current_time is a datetime object
            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            
            # Calculate time window
            start_time = current_time - timedelta(hours=36) # 1.5 days in the past
            end_time = current_time + timedelta(hours=168) # 7 days in the future
            
            query = """
                SELECT 
                    game_id,
                    external_id,
                    event_start_date,
                    team_a,
                    team_b,
                    team_a_odds,
                    team_b_odds,
                    tie_odds,
                    outcome,
                    can_tie,
                    sport,
                    league,
                    create_date,
                    last_update_date,
                    active
                FROM game_data
                WHERE event_start_date >= :start_date
                AND event_start_date <= :end_date
            """
            
            # Pass parameters as a dictionary
            params = {
                "start_date": start_time.isoformat(),
                "end_date": end_time.isoformat()
            }
            
            bt.logging.debug(f"Querying games between {start_time} and {end_time}")
            rows = await self.db_manager.fetch_all(query, params)
            
            if not rows:
                return {}
            
            # Process the results into a dictionary
            gamedata_dict = {}
            for row in rows:
                game_id = row['external_id']
                gamedata_dict[game_id] = {
                    'game_id': game_id,
                    'external_id': game_id,
                    'event_start_date': row['event_start_date'],
                    'team_a': row['team_a'],
                    'team_b': row['team_b'],
                    'team_a_odds': row['team_a_odds'],
                    'team_b_odds': row['team_b_odds'],
                    'tie_odds': row['tie_odds'],
                    'outcome': row['outcome'],
                    'sport': row['sport'],
                    'league': row['league'],
                    'create_date': row['create_date'],
                    'last_update_date': row['last_update_date'],
                    'active': row['active'],
                    'can_tie': row['can_tie']
                }
            
            return gamedata_dict
            
        except Exception as e:
            bt.logging.error(f"Error querying and processing game data: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return {}

    async def validate_prediction(self, miner_uid: int, prediction_id: str, prediction_data: dict) -> tuple[bool, str, str]:
        """
        Validates a prediction before insertion, including checking for duplicates.
        
        Args:
            miner_uid (int): The miner's UID
            prediction_id (str): Unique identifier for the prediction
            prediction_data (dict): The prediction data to validate
            
        Returns:
            tuple: (is_valid, message, validation_type)
        """
        try:
            # First check if prediction already exists
            existing_prediction = await self.db_manager.fetch_one(
                "SELECT prediction_id FROM predictions WHERE prediction_id = ?",
                (prediction_id,)
            )
            
            if existing_prediction:
                return False, f"Prediction {prediction_id} already exists in database", 'duplicate_prediction'
            
            # Check if miner UID is in processed UIDs
            if int(miner_uid) not in self.processed_uids:
                return False, "UID not in processed_uids", 'uid_not_processed'

            # Verify miner exists in metagraph
            try:
                hotkey = self.metagraph.hotkeys[int(miner_uid)]
            except IndexError:
                return False, "Invalid miner UID - not found in metagraph", 'invalid_uid'

            # Validate wager amount
            try:
                wager = float(prediction_data.get('wager', 0))
                if wager <= 0:
                    return False, "Prediction with non-positive wager - nice try", 'invalid_wager'
            except (ValueError, TypeError):
                return False, "Invalid wager value", 'invalid_wager'

            # Check daily wager limit
            current_time = datetime.now(timezone.utc)
            current_total_wager = await self.db_manager.fetch_one(
                "SELECT COALESCE(SUM(wager), 0) as total FROM predictions WHERE miner_uid = ? AND DATE(prediction_date) = DATE(?)",
                (miner_uid, current_time),
            )
            current_total_wager = float(current_total_wager['total'])
            if current_total_wager + wager > 1000:
                return False, f"Prediction would exceed daily limit (Current: ${current_total_wager:.2f}, Attempted: ${wager:.2f})", 'daily_limit_exceeded'

            # Validate game exists and get game data
            query = """
                SELECT sport, league, event_start_date, team_a, team_b, 
                       team_a_odds, team_b_odds, tie_odds, outcome 
                FROM game_data 
                WHERE external_id = ?
            """
            game = await self.db_manager.fetch_one(query, (prediction_data.get('game_id'),))
            
            if not game:
                return False, "Game not found in validator game_data", 'game_not_found'

            # Check if game has started
            if current_time >= datetime.fromisoformat(game['event_start_date']).replace(tzinfo=timezone.utc):
                return False, "Game has already started", 'game_started'

            # Validate predicted outcome
            predicted_outcome = prediction_data.get('predicted_outcome')
            if predicted_outcome == game['team_a']:
                numeric_outcome = 0
            elif predicted_outcome == game['team_b']:
                numeric_outcome = 1
            elif str(predicted_outcome).lower() == "tie":
                numeric_outcome = 2
            else:
                return False, f"Invalid predicted_outcome: {predicted_outcome}", 'invalid_outcome'

            # Validate odds
            outcome_to_odds = {
                0: game['team_a_odds'],
                1: game['team_b_odds'],
                2: game['tie_odds']
            }
            predicted_odds = outcome_to_odds.get(numeric_outcome)
            if predicted_odds is None:
                return False, "Invalid odds for predicted outcome", 'invalid_odds'

            # Optional: Validate confidence score format if provided
            confidence_score = prediction_data.get('confidence_score')
            if confidence_score is not None:
                try:
                    confidence = float(confidence_score)
                    if not 0 <= confidence <= 1:
                        return False, "If provided, confidence score must be between 0 and 1", 'invalid_confidence'
                except (ValueError, TypeError):
                    return False, "Invalid confidence score format", 'invalid_confidence'

            # Model name is optional - no validation needed

            return True, "Prediction validated successfully", 'successful'

        except Exception as e:
            bt.logging.error(f"Error validating prediction: {e}")
            return False, f"Validation error: {str(e)}", 'other_errors'

