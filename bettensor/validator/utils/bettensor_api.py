# File: bettensor/utils/bettensor_api.py

import requests
from datetime import datetime, timedelta, timezone
import bittensor as bt
import json
import os
import sqlite3


class BettensorAPIClient:
    def __init__(self, last_update_file='last_update.json', db_path='data/validator.db'):
        self.base_url = "https://dev-bettensor-api.azurewebsites.net/Games/TeamGames/Search"
        self.last_update_file = last_update_file
        self.last_update_date = self.load_last_update_date()
        self.db_path = db_path
        self.create_table()

    def create_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_data (
                id TEXT PRIMARY KEY,
                teamA TEXT,
                teamB TEXT,
                sport TEXT,
                league TEXT,
                externalId TEXT,
                createDate TEXT,
                lastUpdateDate TEXT,
                eventStartDate TEXT,
                active INTEGER,
                outcome TEXT,
                teamAodds REAL,
                teamBodds REAL,
                tieOdds REAL,
                canTie BOOLEAN
            )
        ''')
        conn.commit()
        conn.close()

    def load_last_update_date(self):
        if os.path.exists(self.last_update_file):
            with open(self.last_update_file, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['last_update_date'])
        return datetime.utcnow() - timedelta(days=7)  # Default to a week ago if no file exists

    def save_last_update_date(self, date):
        with open(self.last_update_file, 'w') as f:
            json.dump({'last_update_date': date.isoformat()}, f)

    def get_games(self, last_update_date=None, items_per_page=100):
        if last_update_date is None:
            last_update_date = self.last_update_date

        all_games = []
        page_index = 0

        while True:
            games = self._get_games_page(last_update_date, items_per_page, page_index)
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

    def _process_games(self, games):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for game in games:
            external_id = game.get('externalId')
            
            # Check if the game exists
            cursor.execute("SELECT 1 FROM game_data WHERE externalId = ?", (external_id,))
            exists = cursor.fetchone()

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
                # Update existing game
                self._update_game(cursor, game, active)
            else:
                # Insert new game
                self._insert_game(cursor, game, active)

        conn.commit()
        conn.close()

    def _update_game(self, cursor, game, active):
        outcome = self._get_numeric_outcome(game)
        
        cursor.execute("""
            UPDATE game_data
            SET teamAodds = ?, teamBodds = ?, tieOdds = ?, lastUpdateDate = ?, active = ?
            WHERE externalId = ?
        """, (
            game.get('teamAOdds'),
            game.get('teamBOdds'),
            game.get('drawOdds'),
            datetime.now(timezone.utc).isoformat(),
            active,
            game.get('externalId')
        ))
        bt.logging.debug(f"Updated game {game.get('externalId')} with active={active}")



    def _insert_game(self, cursor, game, active):
        outcome = self._get_numeric_outcome(game)
        
        cursor.execute("""
            INSERT INTO game_data (
                id, teamA, teamB, sport, league, externalId, createDate, lastUpdateDate,
                eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds, canTie
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(game.get('id')),
            game.get('teamA'),
            game.get('teamB'),
            game.get('sport'),
            game.get('league'),
            game.get('externalId'),
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
            game.get('date'),
            active,
            'Unfinished' if outcome is None else outcome,
            game.get('teamAOdds'),
            game.get('teamBOdds'),
            game.get('drawOdds'),
            game.get('canDraw', True)
        ))
        bt.logging.debug(f"Inserted new game {game.get('externalId')} with active={active}")

    def _get_numeric_outcome(self, game):
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
            bt.logging.warning(f"Unknown outcome {outcome} for game {game.get('externalId')}")
            return None

    def _get_games_page(self, last_update_date, items_per_page, page_index):
        params = {
            "PageIndex": page_index,
            "ItemsPerPage": items_per_page,
            "SortOrder": "StartDate",
            "LastUpdateDate": last_update_date.isoformat(),
            "LeagueFilter": "true"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            
            return data
        except requests.RequestException as e:
            bt.logging.error(f"Error fetching games from API: {e}")
            return None

    def transform_game_data(self, game):
        return {
            "home": game.get("teamA"),
            "away": game.get("teamB"),
            "game_id": game.get("externalId"),
            "date": game.get("date"),
            "odds": {
                "average_home_odds": game.get("teamAOdds"),
                "average_away_odds": game.get("teamBOdds"),
                "average_tie_odds": game.get("drawOdds")
            },
            "sport": game.get("sport").lower(),
            "league": game.get("league")
        }

