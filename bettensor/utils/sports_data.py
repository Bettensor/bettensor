import requests
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from dateutil import parser
import sqlite3
import bittensor as bt
import os

from .api_client import APIClient
from .bettensor_api import BettensorAPIClient


class SportsData:
    def __init__(self, db_name="data/validator.db", use_bt_api=False):
        self.db_name = db_name
        self.use_bt_api = use_bt_api
        self.rapid_api_key = os.getenv("RAPID_API_KEY")
        self.bet365_api_key = os.getenv("BET365_API_KEY")
        self.api_client = APIClient(self.rapid_api_key, self.bet365_api_key)
        self.bettensor_api_client = BettensorAPIClient()
        self.all_games = []
        self.api_hosts = {
            "baseball": "api-baseball.p.rapidapi.com",
            "soccer": "api-football-v1.p.rapidapi.com",
            "nfl": "api.b365api.com"
        }

    def create_database(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS game_data (
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
                    )"""
        )
        conn.commit()
        conn.close()

    def insert_into_database(self, game_data):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute(
            """INSERT INTO game_data (id, teamA, teamB, sport, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds, canTie)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            game_data,
        )
        conn.commit()
        conn.close()

    def update_odds_in_database(self, externalId, teamAodds, teamBodds, tieOdds=None):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        if tieOdds is not None:
            c.execute(
                """UPDATE game_data
                         SET teamAodds = ?, teamBodds = ?, tieOdds = ?, lastUpdateDate = ?
                         WHERE externalId = ?""",
                (
                    teamAodds,
                    teamBodds,
                    tieOdds,
                    datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                    externalId,
                ),
            )
        else:
            c.execute(
                """UPDATE game_data
                         SET teamAodds = ?, teamBodds = ?, lastUpdateDate = ?
                         WHERE externalId = ?""",
                (
                    teamAodds,
                    teamBodds,
                    datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                    externalId,
                ),
            )
        conn.commit()
        conn.close()

    def external_id_exists(self, externalId):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute(
            """SELECT 1 FROM game_data WHERE externalId = ? LIMIT 1""", (externalId,)
        )
        exists = c.fetchone() is not None
        conn.close()
        return exists

    def get_multiple_game_data(self, sports_config):
        all_games = []

        if self.use_bt_api:
            bt.logging.info("Fetching games from BettensorAPI. This will take a while on first run.")
            games = self.bettensor_api_client.get_games()
            if games:
                all_games = [self.bettensor_api_client.transform_game_data(game) for game in games]
            else:
                bt.logging.info("No games to update.")
        else:
            for sport, leagues in sports_config.items():
                for league_info in leagues:
                    league = league_info['id']
                    season = league_info.get('season', '2024')
                    try:
                        if sport == "nfl":
                            games = self.api_client.process_nfl_games()
                        else:
                            games = self.get_game_data(sport=sport, league=league, season=season)
                        all_games.extend(games)
                        
                        # Insert or update games in the database
                        self.insert_or_update_games(games)
                        
                    except Exception as e:
                        bt.logging.error(f"Error fetching data for {sport}, league {league}: {e}")


        bt.logging.debug(f"Initially fetched {len(all_games)} games")
        
        filtered_games = self.filter_games(all_games)

        bt.logging.info(f"Filtered {len(all_games) - len(filtered_games)} games out of {len(all_games)} total games")
        self.all_games = filtered_games

        return filtered_games

    def filter_games(self, games):
        return [
            game for game in games
            if (game["odds"]["average_home_odds"] is not None and
                game["odds"]["average_away_odds"] is not None and
                game["odds"]["average_home_odds"] >= 1.05 and
                game["odds"]["average_away_odds"] >= 1.05 and
                not (game["odds"]["average_home_odds"] == 1.5 and
                     game["odds"]["average_away_odds"] == 3.0 and
                     game["odds"].get("average_tie_odds") == 1.5))
        ]

    def insert_or_update_games(self, games):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        for game in games:
            game_id = str(uuid.uuid4())
            external_id = game['game_id']
            team_a = game['home']
            team_b = game['away']
            sport = game['sport']
            league = game['league']
            create_date = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            last_update_date = create_date
            event_start_date = game['date']
            active = 1
            outcome = "Unfinished"
            team_a_odds = game['odds']['average_home_odds']
            team_b_odds = game['odds']['average_away_odds']
            
            # Set tie_odds to 0.0 for NFL games, otherwise use the value from the game data or 0
            tie_odds = 0.0 if sport.lower() == "nfl" else game['odds'].get('average_tie_odds', 0)
            
            can_tie = sport.lower() == "soccer"

            # Check if the game already exists
            c.execute("SELECT id FROM game_data WHERE externalId = ?", (external_id,))
            existing_game = c.fetchone()

            if existing_game:
                # Update existing game
                c.execute("""
                    UPDATE game_data
                    SET teamAodds = ?, teamBodds = ?, tieOdds = ?, lastUpdateDate = ?
                    WHERE externalId = ?
                """, (team_a_odds, team_b_odds, tie_odds, last_update_date, external_id))
                bt.logging.debug(f"Updated game {external_id} in database")
            else:
                # Insert new game
                c.execute("""
                    INSERT INTO game_data (id, teamA, teamB, sport, league, externalId, createDate, lastUpdateDate,
                                        eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds, canTie)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id, team_a, team_b, sport, league, external_id, create_date, last_update_date,
                    event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie))
                bt.logging.debug(f"Inserted new game {external_id} into database")

        conn.commit()
        conn.close()
        bt.logging.info(f"Inserted or updated {len(games)} games in the database")

    def get_game_data(self, sport, league="1", season="2024"):
        bt.logging.trace(f"Getting game data for sport: {sport}, league: {league}, season: {season}")
        
        start_date = datetime.utcnow().date()
        end_date = start_date + timedelta(days=6)  # 7 days total
        all_games = []

        if sport == "soccer":
            url = f"https://{self.api_hosts[sport]}/v3/fixtures"
            querystring = {
                "league": league,
                "season": season,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            }
            all_games.extend(self._fetch_games(url, querystring, sport))
        elif sport == "baseball":
            url = f"https://{self.api_hosts[sport]}/games"
            for single_date in (start_date + timedelta(n) for n in range(7)):
                querystring = {
                    "league": league,
                    "season": season,
                    "date": single_date.strftime("%Y-%m-%d")
                }
                all_games.extend(self._fetch_games(url, querystring, sport))
        elif sport == "nfl":
            all_games.extend(self._fetch_nfl_games())

        bt.logging.debug(f"Initially fetched {len(all_games)} games for {sport}, league {league}")

        # Filter games with odds less than 1.05; ensure odds are not None; exclude false odds of 1.5/3.0/1.5
        filtered_games = self.filter_games(all_games)

        bt.logging.info(f"Filtered {len(all_games) - len(filtered_games)} games out of {len(all_games)} total games for {sport}")

        # Insert or update the data in the database
        self.insert_or_update_games(filtered_games)

        return filtered_games

    def get_game_data(self, sport, league="1", season="2024"):
        bt.logging.trace(f"Getting game data for sport: {sport}, league: {league}, season: {season}")
        
        start_date = datetime.utcnow().date()
        end_date = start_date + timedelta(days=6)  # 7 days total
        all_games = []

        if sport == "soccer":
            url = f"https://{self.api_hosts[sport]}/v3/fixtures"
            querystring = {
                "league": league,
                "season": season,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            }
            all_games.extend(self._fetch_games(url, querystring, sport))
        elif sport == "baseball":
            url = f"https://{self.api_hosts[sport]}/games"
            for single_date in (start_date + timedelta(n) for n in range(7)):
                querystring = {
                    "league": league,
                    "season": season,
                    "date": single_date.strftime("%Y-%m-%d")
                }
                all_games.extend(self._fetch_games(url, querystring, sport))

        bt.logging.debug(f"Initially fetched {len(all_games)} games for {sport}, league {league}")

        # Filter games with odds less than 1.05; ensure odds are not None; exclude false odds of 1.5/3.0/1.5
        filtered_games = []
        for game in all_games:
            if game["odds"]["average_home_odds"] is None or game["odds"]["average_away_odds"] is None:
                continue
            
            if game["odds"]["average_home_odds"] < 1.05 or game["odds"]["average_away_odds"] < 1.05:
                continue
            
            if (game["odds"]["average_home_odds"] == 1.5 and
                game["odds"]["average_away_odds"] == 3.0 and
                game["odds"].get("average_tie_odds") == 1.5):
                continue
            
            filtered_games.append(game)

        bt.logging.info(f"Filtered {len(all_games) - len(filtered_games)} games out of {len(all_games)} total games")
        all_games = filtered_games

        # Append the fetched games to the overall all_games list
        self.all_games.extend(all_games)

        # Insert or update the data in the database
        for game in all_games:
            try:
                externalId = game["game_id"]
                teamAodds = game["odds"]["average_home_odds"]
                teamBodds = game["odds"]["average_away_odds"]
                tieOdds = game["odds"].get("average_tie_odds") if sport == "soccer" else None

                if self.external_id_exists(externalId):
                    self.update_odds_in_database(externalId, teamAodds, teamBodds, tieOdds)
                else:
                    game_id = str(uuid.uuid4())
                    teamA = game["home"]
                    teamB = game["away"]
                    sport_type = sport
                    createDate = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                    lastUpdateDate = createDate
                    eventStartDate = game["date"]
                    active = 0 if parser.isoparse(eventStartDate) > datetime.utcnow().replace(tzinfo=timezone.utc) else 1
                    outcome = "Unfinished"
                    canTie = sport == "soccer"

                    game_data = (
                        game_id, teamA, teamB, sport_type, league, externalId, createDate, lastUpdateDate,
                        eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds if canTie else 0, canTie,
                    )
                    self.insert_into_database(game_data)
            except KeyError as e:
                bt.logging.error(f"Key Error during database insertion: {e} in game data: {game}")
            except Exception as e:
                bt.logging.error(f"Unexpected error during database insertion: {e}")

        return all_games

    def get_upcoming_events(self):
        url = f"https://api.b365api.com/v1/bet365/upcoming?sport_id=12&token={self.bet365_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            bt.logging.error("Failed to fetch upcoming events")
            return None

    def _fetch_nfl_games(self):
        api_response = self.get_upcoming_events()
        if not api_response:
            return []

        nfl_games = self.get_nfl_games(api_response)
        games_with_odds = []

        for game in nfl_games:
            bt.logging.debug(f"Fetching odds for {game['away']} @ {game['home']} ({game['time']})")
            odds = self.get_odds(game['id'])
            if odds:
                moneyline_odds = self.get_moneyline_odds(odds, game['home'], game['away'])
                games_with_odds.append({
                    "home": game['home'],
                    "away": game['away'],
                    "game_id": str(game['id']),
                    "date": game['time'],
                    "odds": {
                        "average_home_odds": float(moneyline_odds[game['home']]['odds']),
                        "average_away_odds": float(moneyline_odds[game['away']]['odds']),
                        "average_tie_odds": None
                    },
                    "sport": "nfl",
                    "league": "12",
                })

        return games_with_odds

    def get_nfl_games(self, api_response):
        nfl_games = []
        for game in api_response['results']:
            if game['league']['name'] == "NFL":
                nfl_games.append({
                    'id': game['id'],
                    'home': game['home']['name'],
                    'away': game['away']['name'],
                    'time': datetime.fromtimestamp(int(game['time'])).replace(tzinfo=timezone.utc).isoformat()
                })
        return nfl_games

    def get_odds(self, event_id):
        url = f"https://api.b365api.com/v3/bet365/prematch?token={self.bet365_api_key}&FI={event_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            bt.logging.error(f"Failed to fetch odds for event {event_id}")
            return None

    def get_moneyline_odds(self, odds_data, home_team, away_team):
        moneyline_odds = {}
        for market in odds_data['results'][0]['main']['sp'].values():
            if market['name'] == 'Game Lines':
                for odd in market['odds']:
                    if odd['name'] == 'Money Line':
                        team = home_team if odd['header'] == "1" else away_team
                        moneyline_odds[team] = {
                            'odds': odd['odds'],
                            'implied_probability': f"{(1 / float(odd['odds'])) * 100:.2f}%"
                        }
        return moneyline_odds

    def _fetch_games(self, url, querystring, sport):
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": self.api_hosts[sport],
        }

        try:
            bt.logging.debug(f"Sending request to {url} with params: {querystring}")
            response = requests.get(url, headers=headers, params=querystring)
            bt.logging.debug(f"Response status code: {response.status_code}")
            
            if response.status_code == 429:
                bt.logging.warning("Rate limit exceeded. Waiting for 60 seconds before retrying.")
                time.sleep(60)
                return self._fetch_games(url, querystring, sport)  # Retry after waiting
            
            response.raise_for_status()
            games = response.json()
            
            if "response" in games:
                games_list = [
                    {
                        "home": i["teams"]["home"]["name"],
                        "away": i["teams"]["away"]["name"],
                        "game_id": i["fixture"]["id"] if sport == "soccer" else i["id"],
                        "date": i["fixture"]["date"] if sport == "soccer" else i["date"],
                        "odds": self.get_game_odds(i["fixture"]["id"] if sport == "soccer" else i["id"], sport)
                    }
                    for i in games["response"]
                ]
                bt.logging.debug(f"Fetched {len(games_list)} games for {sport}")
                return games_list
            else:
                bt.logging.warning(f"Unexpected response format: {games}")
                return []
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"HTTP Request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            bt.logging.error(f"JSON Decode Error: {e}")
            return []

    def get_game_odds(self, game_id, sport):
        if sport == "soccer":
            url = f"https://{self.api_hosts[sport]}/v3/odds"
            querystring = {"fixture": game_id}
        elif sport == "baseball":
            url = f"https://{self.api_hosts[sport]}/odds"
            querystring = {"game": game_id}

        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": self.api_hosts[sport],
        }

        try:
            bt.logging.debug(f"Fetching odds for game {game_id} in {sport}")
            response = requests.get(url, headers=headers, params=querystring)
            bt.logging.debug(f"Odds response status code: {response.status_code}")
            
            if response.status_code == 429:
                bt.logging.warning("Rate limit exceeded while fetching odds. Waiting for 60 seconds before retrying.")
                time.sleep(60)
                return self.get_game_odds(game_id, sport)  # Retry after waiting
            
            response.raise_for_status()  # Ensure we handle HTTP errors properly

            odds_data = response.json()

            # Initialize totals and count
            total_home_odds = 0
            total_away_odds = 0
            total_tie_odds = 0
            count = 0

            # Ensure response is not empty and contains the necessary data
            if odds_data.get("response") and odds_data["response"][0].get("bookmakers"):
                bookmakers = odds_data["response"][0]["bookmakers"]
                for bookmaker in bookmakers:
                    for bet in bookmaker.get("bets", []):
                        if (
                            sport == "soccer"
                            and bet["name"] == "Match Winner"
                            and len(bet["values"]) >= 3
                        ):
                            odds_dict = {odd["value"]: float(odd["odd"]) for odd in bet["values"]}
                            home_odds = odds_dict.get("Home")
                            away_odds = odds_dict.get("Away")
                            tie_odds = odds_dict.get("Draw")  # Note: It might be "Draw" instead of "Tie"
                            
                            if home_odds and away_odds and tie_odds:
                                total_home_odds += home_odds
                                total_away_odds += away_odds
                                total_tie_odds += tie_odds
                                count += 1
                            else:
                                bt.logging.warning(f"Missing odds for game {game_id} in bookmaker {bookmaker['name']}")
                        elif (
                            sport == "baseball"
                            and bet["name"] == "Home/Away"
                            and len(bet["values"]) >= 2
                        ):
                            odds_dict = {odd["value"]: float(odd["odd"]) for odd in bet["values"]}
                            home_odds = odds_dict.get("Home")
                            away_odds = odds_dict.get("Away")
                            
                            if home_odds and away_odds:
                                total_home_odds += home_odds
                                total_away_odds += away_odds
                                count += 1
                            else:
                                bt.logging.warning(f"Missing odds for game {game_id} in bookmaker {bookmaker['name']}")
                # Calculate average odds if count is greater than 0
                if count > 0:
                    avg_home_odds = round(total_home_odds / count, 2) if total_home_odds else None
                    avg_away_odds = round(total_away_odds / count, 2) if total_away_odds else None
                    avg_tie_odds = round(total_tie_odds / count, 2) if sport == "soccer" and total_tie_odds else None
                    
                    return {
                        "average_home_odds": avg_home_odds,
                        "average_away_odds": avg_away_odds,
                        "average_tie_odds": avg_tie_odds if sport == "soccer" else None,
                    }

            # Return None values if no data is available
            return {
                "average_home_odds": None,
                "average_away_odds": None,
                "average_tie_odds": None,
            }
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Error fetching odds for game {game_id} in {sport}: {e}")
            return {
                "average_home_odds": None,
                "average_away_odds": None,
                "average_tie_odds": None,
            }