import os
import json
import time
from typing import List
import uuid
import sqlite3
import requests
import bittensor as bt
from dateutil import parser
from .sports_config import sports_config
from datetime import datetime, timedelta, timezone
from ..scoring.entropy_system import EntropySystem
from .bettensor_api_client import BettensorAPIClient
from bettensor.validator.utils.database.database_manager import DatabaseManager


class SportsData:
    """
    SportsData class is responsible for fetching and updating sports data from either BettensorAPI or external API.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        entropy_system: EntropySystem,
        api_client,
    ):
        self.db_manager = db_manager
        self.entropy_system = entropy_system
        self.api_client = api_client
        self.all_games = []

    def fetch_and_update_game_data(self, last_api_call):
        try:
            all_games = self.api_client.fetch_all_game_data(last_api_call)
            inserted_ids = self.insert_or_update_games(all_games)
            self.update_predictions_with_payouts(inserted_ids)
            self.all_games = all_games
            return inserted_ids
        except Exception as e:
            bt.logging.error(f"Error fetching game data: {str(e)}")
            raise  # Re-raise the exception

    def insert_or_update_games(self, games):
        if not games:
            bt.logging.info("No games to insert or update")
            return []

        self.db_manager.begin_transaction()
        external_ids = []
        try:
            for game in games:
                game_id = str(uuid.uuid4())
                external_id = game["externalId"]
                team_a = game["teamA"]
                team_b = game["teamB"]
                sport = game["sport"]
                league = game["league"]
                create_date = datetime.now(timezone.utc).isoformat()
                last_update_date = create_date
                event_start_date = game["date"]
                active = 1
                outcome = game["outcome"]

                team_a_odds = game["teamAOdds"]
                team_b_odds = game["teamBOdds"]
                tie_odds = game["drawOdds"]

                can_tie = sport.lower() == "soccer"

                if sport.lower() == "football":
                    can_tie = False
                    tie_odds = 0.0

                # Ensure event_start_date is timezone aware
                event_start_time = datetime.fromisoformat(event_start_date)
                if event_start_time.tzinfo is None:
                    event_start_time = event_start_time.replace(tzinfo=timezone.utc)

                # Convert outcomes to numeric
                if outcome == "TeamAWin":
                    outcome = 0
                elif outcome == "TeamBWin":
                    outcome = 1
                elif outcome == "Draw" and datetime.now(timezone.utc) - event_start_time > timedelta(hours=4):
                    outcome = 2
                else:
                    outcome = 3

                # Set active to 0 if outcome is not "Unfinished" and more than 4 hours have passed
                if outcome != 3 and (datetime.now(timezone.utc) - event_start_time) > timedelta(hours=4):
                    active = 0

                self.db_manager.execute_query(
                    """
                    INSERT INTO game_data (game_id, team_a, team_b, sport, league, external_id, create_date, last_update_date,
                                        event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(external_id) DO UPDATE SET
                        team_a_odds = excluded.team_a_odds,
                        team_b_odds = excluded.team_b_odds,
                        tie_odds = excluded.tie_odds,
                        event_start_date = excluded.event_start_date,
                        active = excluded.active,
                        outcome = excluded.outcome,
                        last_update_date = excluded.last_update_date
                    """,
                    (
                        game_id,
                        team_a,
                        team_b,
                        sport,
                        league,
                        external_id,
                        create_date,
                        last_update_date,
                        event_start_date,
                        active,
                        outcome,
                        team_a_odds,
                        team_b_odds,
                        tie_odds,
                        can_tie,
                    ),
                )
                bt.logging.debug(f"Upserted game {external_id} in database")
                external_ids.append(external_id)
                outcomes = 3 if can_tie else 2
                self.entropy_system.add_new_game(external_id, outcomes, [team_a_odds, team_b_odds, tie_odds])
                self.entropy_system.save_state("entropy_system_state.json")

            self.db_manager.commit_transaction()
            bt.logging.info(f"Inserted or updated {len(games)} games in the database")
            return external_ids

        except Exception as e:
            self.db_manager.rollback_transaction()
            bt.logging.error(f"Error inserting or updating games: {e}")
            raise

    def prepare_game_data_for_entropy(self, games):
        game_data = []
        for game in games:
            if self.is_bettensor_api():
                game_data.append(
                    {
                        "id": game["externalId"],
                        "predictions": {},  # No predictions yet for new games
                        "current_odds": [
                            game["teamAOdds"],
                            game["teamBOdds"],
                            game.get("tieOdds", 0.0),
                        ],
                    }
                )
            else:
                game_data.append(
                    {
                        "id": game["game_id"],
                        "predictions": {},  # No predictions yet for new games
                        "current_odds": [
                            game["odds"]["average_home_odds"],
                            game["odds"]["average_away_odds"],
                            game["odds"].get("average_tie_odds", 0.0),
                        ],
                    }
                )
        return game_data

    def update_predictions_with_payouts(self, external_ids):
        """
        Retrieve all predictions associated with the provided external IDs, determine if each prediction won,
        calculate payouts, and update the predictions in the database.

        Args:
            external_ids (List[str]): List of external_id's of the games that were inserted/updated.
        """
        try:
            if not external_ids:
                bt.logging.info("No external IDs provided for updating predictions.")
                return

            # Fetch outcomes for the given external_ids
            query = """
                SELECT external_id, outcome
                FROM game_data
                WHERE external_id IN ({seq})
            """.format(
                seq=",".join(["?"] * len(external_ids))
            )
            game_outcomes = self.db_manager.fetch_all(query, tuple(external_ids))
            game_outcome_map = {
                external_id: outcome for external_id, outcome in game_outcomes
            }

            bt.logging.info(f"Fetched outcomes for {len(game_outcomes)} games.")

            # Fetch all predictions associated with the external_ids
            query = """
                SELECT prediction_id, miner_uid, game_id, predicted_outcome, predicted_odds, wager
                FROM predictions
                WHERE game_id IN ({seq}) 
            """.format(
                seq=",".join(["?"] * len(external_ids))
            )
            predictions = self.db_manager.fetch_all(query, tuple(external_ids))

            bt.logging.info(f"Fetched {len(predictions)} predictions to process.")

            for prediction in predictions:
                
                (
                    prediction_id,
                    miner_uid,
                    game_id,
                    predicted_outcome,
                    predicted_odds,
                    wager,
                ) = prediction
                if game_id == "game_id":
                    continue
                actual_outcome = game_outcome_map.get(game_id)

                if actual_outcome is None:
                    bt.logging.warning(
                        f"No outcome found for game {game_id}. Skipping prediction {prediction_id}."
                    )
                    continue

                is_winner = predicted_outcome == actual_outcome
                payout = wager * predicted_odds if is_winner else 0

                update_query = """
                    UPDATE predictions
                    SET result = ?, payout = ?, processed = 1
                    WHERE prediction_id = ?
                """
                self.db_manager.execute_query(
                    update_query, (is_winner, payout, prediction_id)
                )

                if is_winner:
                    bt.logging.info(
                        f"Prediction {prediction_id}: Miner {miner_uid} won. Payout: {payout}"
                    )
                else:
                    bt.logging.info(
                        f"Prediction {prediction_id}: Miner {miner_uid} lost."
                    )

            self.db_manager.commit_transaction()
            bt.logging.info(
                "All predictions have been processed and updated successfully."
            )

        except Exception as e:
            self.db_manager.rollback_transaction()
            bt.logging.error(f"Error in update_predictions_with_payouts: {e}")
            raise

    




