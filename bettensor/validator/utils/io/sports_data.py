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
    '''
    SportsData class is responsible for fetching and updating sports data from either BettensorAPI or external API.
    '''
    def __init__(self, db_manager: DatabaseManager, entropy_system: EntropySystem, use_bt_api=False):
        self.db_manager = db_manager
        self.entropy_system = entropy_system
        self.use_bt_api = use_bt_api
        
        #initialize api client depending on use_bt_api flag
        if self.use_bt_api:
            self.api_client = BettensorAPIClient()
        else:
            self.api_client = ExternalAPIClient()
            
        self.all_games = []

    def fetch_and_update_game_data(self):
        '''
        Single method to fetch and update game data from either BettensorAPI or external API.
        This method will be called from the validator.py file.
        
        '''
        all_games = self.api_client.fetch_all_game_data()
        self.insert_or_update_games(all_games)
        self.all_games = all_games

    def insert_or_update_games(self, games):
        self.db_manager.begin_transaction()
        try:
            for game in games:
                game_id = str(uuid.uuid4())
                external_id = game['game_id']
                team_a = game['home']
                team_b = game['away']
                sport = game['sport']
                league = game['league']
                create_date = datetime.now(timezone.utc).isoformat()
                last_update_date = create_date
                event_start_date = game['date']
                active = 1
                outcome = "Unfinished"
                team_a_odds = game['odds']['average_home_odds']
                team_b_odds = game['odds']['average_away_odds']
                
                tie_odds = 0.0 if sport.lower() == "football" else game['odds'].get('average_tie_odds', 0)
                can_tie = sport.lower() == "soccer"

                self.db_manager.execute_query(
                    """
                    INSERT INTO game_data (id, team_a, team_b, sport, league, external_id, create_date, last_update_date,
                                        event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(external_id) DO UPDATE SET
                        team_a_odds = excluded.team_a_odds,
                        team_b_odds = excluded.team_b_odds,
                        tie_odds = excluded.tie_odds,
                        last_update_date = excluded.last_update_date
                    """, 
                    (game_id, team_a, team_b, sport, league, external_id, create_date, last_update_date,
                    event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie)
                )
                bt.logging.debug(f"Upserted game {external_id} in database")

            self.db_manager.commit_transaction()
            bt.logging.info(f"Inserted or updated {len(games)} games in the database")

            # After inserting/updating games, update entropy scores
            game_data = self.prepare_game_data_for_entropy(games)
            self.entropy_system.update_ebdr_scores(game_data)
        except Exception as e:
            self.db_manager.rollback_transaction()
            bt.logging.error(f"Error inserting or updating games: {e}")
            raise

    def prepare_game_data_for_entropy(self, games):
        game_data = []
        for game in games:
            game_data.append({
                'id': game['id'],
                'predictions': {},  # No predictions yet for new games
                'current_odds': [
                    game['odds']['average_home_odds'],
                    game['odds']['average_away_odds'],
                    game['odds'].get('average_tie_odds', 0.0)
                ]
            })
        return game_data

