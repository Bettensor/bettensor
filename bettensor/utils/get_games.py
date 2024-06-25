import requests
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from dateutil import parser
import sqlite3

import os
from dotenv import load_dotenv

class GetGames:
    def __init__(self, db_name='games_data.db'):
        # load the .env file
        load_dotenv()
        self.db_name = db_name
        self.api_key = os.getenv("RAPID_API_KEY")
        self.api_hosts = {
            "soccer": "api-football-v1.p.rapidapi.com",
            "baseball": "api-baseball.p.rapidapi.com"
        }
        self.create_database()

    def create_database(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS game_data (
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
                    )''')
        conn.commit()
        conn.close()

    def insert_into_database(self, game_data):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''INSERT INTO game_data (id, teamA, teamB, sport, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds, canTie)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', game_data)
        conn.commit()
        conn.close()

    def update_odds_in_database(self, externalId, teamAodds, teamBodds, tieOdds=None):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        if tieOdds is not None:
            c.execute('''UPDATE game_data
                         SET teamAodds = ?, teamBodds = ?, tieOdds = ?, lastUpdateDate = ?
                         WHERE externalId = ?''', (teamAodds, teamBodds, tieOdds, datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), externalId))
        else:
            c.execute('''UPDATE game_data
                         SET teamAodds = ?, teamBodds = ?, lastUpdateDate = ?
                         WHERE externalId = ?''', (teamAodds, teamBodds, datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), externalId))
        conn.commit()
        conn.close()

    def external_id_exists(self, externalId):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''SELECT 1 FROM game_data WHERE externalId = ? LIMIT 1''', (externalId,))
        exists = c.fetchone() is not None
        conn.close()
        return exists

    def get_game_data(self, sport, leagues, season="2024", utc_offset=0):
        dates = [datetime.utcnow().date() + timedelta(days=i) for i in range(7)]
        all_games = []

        for league in leagues:
            for date in dates:
                date_str = self.adjust_date_for_utc_offset(date, utc_offset_hours).strftime("%Y-%m-%d")
                if sport == "soccer":
                    fixtures_url = f"https://{self.api_hosts[sport]}/v3/fixtures"
                    fixtures_query = {"league": league, "season": season, "date": date_str}
                elif sport == "baseball":
                    fixtures_url = f"https://{self.api_hosts[sport]}/games"
                    fixtures_query = {"league": league, "season": season, "date": date_str}

                headers = {
                    "X-RapidAPI-Key": self.api_key,
                    "X-RapidAPI-Host": self.api_hosts[sport]
                }

                try:
                    response = requests.get(fixtures_url, headers=headers, params=fixtures_query)
                    response.raise_for_status()
                    fixtures = response.json()
                except requests.exceptions.RequestException as e:
                    print(f"HTTP Request failed: {e}")
                    continue
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    continue

                if 'response' not in fixtures:
                    print(f"Unexpected response format: {fixtures}")
                    continue

                try:
                    games_list = [{
                        "home": i['teams']['home']['name'],
                        "away": i['teams']['away']['name'],
                        "game_id": i['fixture']['id'] if sport == "soccer" else i['id'],
                        "date": i['fixture']['date'] if sport == "soccer" else i['date'],
                        "venue": i['fixture']['venue']['name'] if sport == "soccer" else None,
                        "odds": self.get_game_odds(i['fixture']['id'] if sport == "soccer" else i['id'], sport)
                    } for i in fixtures['response']]
                except KeyError as e:
                    print(f"Key Error: {e} in fixture data: {i}")
                    continue

                all_games.extend(games_list)

        # Save the extracted data to a JSON file
        with open(f'{sport}_games_data.json', 'w') as f:
            json.dump(all_games, f, indent=4)

        # Insert or update the data in the database
        for game in all_games:
            try:
                externalId = game['game_id']
                teamAodds = game['odds']['average_home_odds']
                teamBodds = game['odds']['average_away_odds']
                tieOdds = game['odds'].get('average_tie_odds') if sport == "soccer" else None

                if self.external_id_exists(externalId):
                    # Update the existing record with new odds
                    self.update_odds_in_database(externalId, teamAodds, teamBodds, tieOdds)
                else:
                    # Insert a new record
                    game_id = str(uuid.uuid4())
                    teamA = game['home']
                    teamB = game['away']
                    sport_type = sport
                    createDate = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                    lastUpdateDate = createDate
                    eventStartDate = game['date']
                    active = 0 if parser.isoparse(eventStartDate) > datetime.utcnow().replace(tzinfo=timezone.utc) else 1
                    outcome = "Unfinished"
                    canTie = sport == "soccer"

                    game_data = (game_id, teamA, teamB, sport_type, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds if canTie else 0, canTie)
                    self.insert_into_database(game_data)
                    bt.logging.info(f"GAME DATA: {game_data}"
            except KeyError as e:
                print(f"Key Error during database insertion: {e} in game data: {game}")
            except Exception as e:
                print(f"Unexpected error during database insertion: {e}")

        return all_games

    def adjust_date_for_utc_offset(self, date, utc_offset_hours):
            return date + timedelta(hours=utc_offset_hours)

    def get_game_odds(self, game_id, sport):
        if sport == "soccer":
            odds_url = f"https://{self.api_hosts[sport]}/v3/odds"
            odds_query = {"fixture": game_id}
        elif sport == "baseball":
            odds_url = f"https://{self.api_hosts[sport]}/odds"
            odds_query = {"game": game_id}

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.api_hosts[sport]
        }

        response = requests.get(odds_url, headers=headers, params=odds_query)
        response.raise_for_status()

        odds_data = response.json()
        total_home_odds = 0
        total_away_odds = 0
        total_tie_odds = 0
        count = 0
        
        if odds_data['response']:
            bookmakers = odds_data['response'][0]['bookmakers']
            for bookmaker in bookmakers:
                for bet in bookmaker['bets']:
                    if bet['name'] in ["Match Winner", "Home/Away"]:
                        if sport == "soccer" and len(bet['values']) >= 3:
                            home_odds = float(bet['values'][0]['odd'])
                            away_odds = float(bet['values'][1]['odd'])
                            tie_odds = float(bet['values'][2]['odd'])
                            total_home_odds += home_odds
                            total_away_odds += away_odds
                            total_tie_odds += tie_odds
                            count += 1
                        elif sport == "baseball" and len(bet['values']) >= 2:
                            home_odds = float(bet['values'][0]['odd'])
                            away_odds = float(bet['values'][1]['odd'])
                            total_home_odds += home_odds
                            total_away_odds += away_odds
                            count += 1

            if count > 0:
                avg_home_odds = round(total_home_odds / count, 2) if total_home_odds != 0 else None
                avg_away_odds = round(total_away_odds / count, 2) if total_away_odds != 0 else None
                avg_tie_odds = round(total_tie_odds / count, 2) if total_tie_odds != 0 else None
                return {"average_home_odds": avg_home_odds, "average_away_odds": avg_away_odds, "average_tie_odds": avg_tie_odds if sport == "soccer" else None}
        
        return {"average_home_odds": None, "average_away_odds": None, "average_tie_odds": None}

    def print_all_data(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''SELECT * FROM game_data''')
        rows = c.fetchall()
        column_names = [description[0] for description in c.description]
        print("\t".join(column_names))
        for row in rows:
            print("\t".join(str(item) for item in row))
        conn.close()

# Example usage:
get_games = GetGames()
soccer_leagues = ["253", "140", "78"]  # MLS, La Liga, Bundesliga
baseball_leagues = ["1"]  # MLB (example)

# Fetch and store soccer data
soccer_games = get_games.get_game_data("soccer", soccer_leagues)

# Fetch and store baseball data
baseball_games = get_games.get_game_data("baseball", baseball_leagues)

# Print all data from the database
get_games.print_all_data()

