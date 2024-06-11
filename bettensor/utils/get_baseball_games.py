import requests
import json
import time
import uuid
from datetime import datetime
import sqlite3

class BaseballData:
    def __init__(self, db_name='games_data.db'):
        self.db_name = db_name
        self.api_key = "b416b1c26dmsh6f20cd13ee1f7ccp11cc1djsnf64975aaacde"
        self.api_host = "api-baseball.p.rapidapi.com"
        self.create_database()

    def create_database(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS game_data (
                        id TEXT PRIMARY KEY,
                        teamA TEXT,
                        teamB TEXT,
                        teamAodds REAL,
                        teamBodds REAL,
                        sport TEXT,
                        league TEXT,
                        externalId TEXT,
                        createDate TEXT,
                        lastUpdateDate TEXT,
                        eventStartDate TEXT,
                        active INTEGER,
                        outcome TEXT
                    )''')
        conn.commit()
        conn.close()

    def insert_into_database(self, game_data):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''INSERT INTO game_data (id, teamA, teamB, teamAodds, teamBodds, sport, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', game_data)
        conn.commit()
        conn.close()

    def get_baseball_data(self, league="1", season="2024", date="2024-05-29"):
        url = "https://api-baseball.p.rapidapi.com/games"

        querystring = {"league": league, "season": season, "date": date}

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.api_host
        }

        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Ensure we handle HTTP errors properly
        games = response.json()

        games_list = [{
            "home": i['teams']['home']['name'],
            "away": i['teams']['away']['name'],
            "game_id": i['id'],
            "date": i['date'],
            "odds": self.get_game_odds(i['id'])  # Get averaged odds for each game
        } for i in games['response']]
        
        # Save the extracted data to a JSON file
        with open('games_data.json', 'w') as f:
            json.dump(games_list, f, indent=4)

        # Insert the data into the database
        for game in games_list:
            game_id = str(uuid.uuid4())
            teamA = game['home']
            teamB = game['away']
            teamAodds = game['odds']['average_home_odds'] if game['odds']['average_home_odds'] is not None else 0.0
            teamBodds = game['odds']['average_away_odds'] if game['odds']['average_away_odds'] is not None else 0.0
            sport = "baseball"
            league = league
            externalId = game['game_id']
            createDate = datetime.utcnow().isoformat()
            lastUpdateDate = createDate
            eventStartDate = game['date']
            active = 1 if datetime.fromisoformat(eventStartDate[:-1]) > datetime.utcnow() else 0
            outcome = ""

            game_data = (game_id, teamA, teamB, teamAodds, teamBodds, sport, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome)
            self.insert_into_database(game_data)

        return games_list

    def get_game_odds(self, game_id):
        url = "https://api-baseball.p.rapidapi.com/odds"

        querystring = {"game": game_id}

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.api_host
        }

        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Ensure we handle HTTP errors properly

        odds_data = response.json()
        total_home_odds = 0
        total_away_odds = 0
        count = 0
        
        if odds_data['results'] > 0:
            bookmakers = odds_data['response'][0]['bookmakers']
            for bookmaker in bookmakers:
                for bet in bookmaker['bets']:
                    if bet['name'] == "Home/Away":
                        if len(bet['values']) >= 2:
                            home_odds = float(bet['values'][0]['odd'])
                            away_odds = float(bet['values'][1]['odd'])
                            total_home_odds += home_odds
                            total_away_odds += away_odds
                            count += 1
                        elif len(bet['values']) == 1:
                            home_odds = float(bet['values'][0]['odd']) if bet['values'][0]['value'] == 'Home' else None
                            away_odds = float(bet['values'][0]['odd']) if bet['values'][0]['value'] == 'Away' else None
                            if home_odds is not None:
                                total_home_odds += home_odds
                                count += 1
                            if away_odds is not None:
                                total_away_odds += away_odds
                                count += 1

            if count > 0:
                avg_home_odds = round(total_home_odds / count, 2) if total_home_odds != 0 else None
                avg_away_odds = round(total_away_odds / count, 2) if total_away_odds != 0 else None
                return {"average_home_odds": avg_home_odds, "average_away_odds": avg_away_odds}
        
        return {"average_home_odds": None, "average_away_odds": None}
