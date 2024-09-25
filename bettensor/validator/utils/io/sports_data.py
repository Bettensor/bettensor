import os
import json
import time
import uuid
import sqlite3
import requests
import bittensor as bt
from dateutil import parser
from .sports_config import sports_config
from .external_api_client import ExternalAPIClient
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
            self.insert_or_update_games(all_games)
            self.all_games = all_games
            return all_games
        except Exception as e:
            bt.logging.error(f"Error fetching game data: {str(e)}")
            raise  # Re-raise the exception

    def insert_or_update_games(self, games):
        if not games:
            bt.logging.info("No games to insert or update")
            return

        self.db_manager.begin_transaction()
        try:
            for game in games:
                game_id = str(uuid.uuid4())
                external_id = game["externalId"] if self.is_bettensor_api() else game["game_id"]
                team_a = game["teamA"] if self.is_bettensor_api() else game["home"]
                team_b = game["teamB"] if self.is_bettensor_api() else game["away"]
                sport = game["sport"]
                league = game["league"]
                create_date = datetime.now(timezone.utc).isoformat()
                last_update_date = create_date
                event_start_date = game["date"]
                active = 1
                outcome = "Unfinished"

                if self.is_bettensor_api():
                    team_a_odds = game["teamAOdds"]
                    team_b_odds = game["teamBOdds"]
                    tie_odds = game.get("tieOdds", 0.0)
                else:
                    team_a_odds = game["odds"]["average_home_odds"]
                    team_b_odds = game["odds"]["average_away_odds"]
                    tie_odds = (
                        0.0
                        if sport.lower() == "football"
                        else game["odds"].get("average_tie_odds", 0)
                    )

                can_tie = sport.lower() == "soccer"

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

            self.db_manager.commit_transaction()
            bt.logging.info(f"Inserted or updated {len(games)} games in the database")

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

    def is_bettensor_api(self):
        return isinstance(self.api_client, BettensorAPIClient)
