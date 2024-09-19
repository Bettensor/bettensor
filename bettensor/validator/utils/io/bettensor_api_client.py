# File: bettensor/utils/bettensor_api.py

import requests
from datetime import datetime, timedelta, timezone
import bittensor as bt
import json
import os
from .base_api_client import BaseAPIClient
from ..database.database_manager import DatabaseManager

class BettensorAPIClient(BaseAPIClient):
    def __init__(self, base_url: str, api_key: str, last_update_file='last_update.json', db_path='data/validator.db'):
        super().__init__(base_url, api_key)

        self.base_url = "https://dev-bettensor-api.azurewebsites.net/"
        self.db_path = db_path
        self.create_table()

    # Existing methods (commented out for reference)
    '''
    def __init__(self, last_update_file='last_update.json', db_path='data/validator.db'):
        
        self.last_update_file = last_update_file
        self.last_update_date = self.load_last_update_date()
        self.db_path = db_path
        self.create_table()
    '''


    # New method using BaseAPIClient
    def get_games_new(self, last_update_date=None, items_per_page=100):
        if last_update_date is None:
            last_update_date = self.last_update_date

        all_games = []
        page_index = 0

        while True:
            games = self._get_games_page_new(last_update_date, items_per_page, page_index)
            if not games:
                break
            all_games.extend(games)
            page_index += 1

        self._process_games(all_games)

        # Update and save the last_update_date on successful response
        new_last_update = datetime.utcnow()
        self.save_last_update_date(new_last_update)
        self.last_update_date = new_last_update

        return all_games

    # Existing method (commented out for reference)
    '''
    def get_games(self, last_update_date=None, items_per_page=100):
        # Existing implementation
    '''

    def _process_games(self, games):
        for game in games:
            external_id = game.get('external_id')
            
            query = "SELECT 1 FROM game_data WHERE external_id = ?"
            exists = self.db_manager.fetchone(query, (external_id,))

            # Get the current time in UTC
            current_time = datetime.now(timezone.utc)

            # Parse the event start date
            event_start_date_str = game.get('date')
            try:
                event_start_date = datetime.fromisoformat(event_start_date_str).replace(tzinfo=timezone.utc)
            except ValueError:
                bt.logging.error(f"Invalid date format for game {external_id}: {event_start_date_str}")
                continue

            # Determine if the game is active
            active = 1 if current_time <= event_start_date else 0

            bt.logging.debug(f"Game {external_id}: current_time={current_time}, event_start_date={event_start_date}, active={active}")

            if exists:
                self._update_game(game, active)
            else:
                self._insert_game(game, active)

    def _update_game(self, game, active):
        query = """
            UPDATE game_data
            SET team_a_odds = ?, team_b_odds = ?, tie_odds = ?, last_update_date = ?, active = ?
            WHERE external_id = ?
        """
        params = (
            game.get('team_a_odds'),
            game.get('team_b_odds'),
            game.get('tie_odds'),
            datetime.now(timezone.utc).isoformat(),
            active,
            game.get('external_id')
        )
        self.db_manager.execute_query(query, params)
        bt.logging.debug(f"Updated game {game.get('external_id')} with active={active}")

    def _insert_game(self, game, active):
        outcome = self._get_numeric_outcome(game)
        query = """
            INSERT INTO game_data (
                id, team_a, team_b, sport, league, external_id, create_date, last_update_date,
                event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            str(game.get('id')),
            game.get('team_a'),
            game.get('team_b'),
            game.get('sport'),
            game.get('league'),
            game.get('external_id'),
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
            game.get('event_start_date'),
            active,
            'Unfinished' if outcome is None else outcome,
            game.get('team_a_odds'),
            game.get('team_b_odds'),
            game.get('tie_odds'),
            game.get('can_tie', True)
        )
        self.db_manager.execute_query(query, params)
        bt.logging.debug(f"Inserted new game {game.get('external_id')} with active={active}")

    def _get_numeric_outcome(self, game):
        # Existing implementation
        outcome = game.get('outcome')
        if outcome is None:
            return 'Unfinished'
        elif outcome == 'TeamAWin':
            return '0'
        elif outcome == 'TeamBWin':
            return '1'
        elif outcome.lower() == 'draw':
            return '2'
        else:
            bt.logging.warning(f"Unknown outcome {outcome} for game {game.get('external_id')}")
            return None

    # New method using BaseAPIClient
    def _get_games_page_new(self, last_update_date, items_per_page, page_index):
        params = {
            "PageIndex": page_index,
            "ItemsPerPage": items_per_page,
            "SortOrder": "StartDate",
            "LastUpdateDate": last_update_date.isoformat(),
            "LeagueFilter": "true"
        }
        
        try:
            data = self.get_data("Games/TeamGames/Search", params)
            return data
        except Exception as e:
            bt.logging.error(f"Error fetching games from API: {e}")
            return None

    # Existing method (commented out for reference)
    '''
    def _get_games_page(self, last_update_date, items_per_page, page_index):
        # Existing implementation
    '''

    def transform_game_data(self, game):
        # Existing implementation
        return {
            "home": game.get("team_a"),
            "away": game.get("team_b"),
            "game_id": game.get("external_id"),
            "date": game.get("date"),
            "odds": {
                "average_home_odds": game.get("team_a_odds"),
                "average_away_odds": game.get("team_b_odds"),
                "average_tie_odds": game.get("tie_odds")
            },
            "sport": game.get("sport").lower(),
            "league": game.get("league")
        }

