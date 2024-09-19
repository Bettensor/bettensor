import os
import json
import time
import uuid
import sqlite3
import requests
import bittensor as bt
from dateutil import parser
from .sports_config import sports_config
from .external_api_client import APIClient
from datetime import datetime, timedelta, timezone
from ..scoring.entropy_system import EntropySystem
from .bettensor_api_client import BettensorAPIClient
from bettensor.validator.utils.database.database_manager import DatabaseManager


class SportsData:
    def __init__(self, db_manager: DatabaseManager, use_bt_api=False):
        self.db_manager = db_manager
        self.use_bt_api = use_bt_api
        self.rapid_api_key = os.getenv("RAPID_API_KEY")
        self.bet365_api_key = os.getenv("BET365_API_KEY")
        self.api_client = APIClient(self.rapid_api_key, self.bet365_api_key)
        self.bettensor_api_client = BettensorAPIClient()
        self.all_games = []
        self.use_bt_api = use_bt_api
        self.sports_config = sports_config
        self.rapid_api_key = os.getenv("RAPID_API_KEY") if not use_bt_api else None
        self.bet365_api_key = os.getenv("BET365_API_KEY") if not use_bt_api else None
        self.external_api_client = APIClient(self.rapid_api_key, self.bet365_api_key)
        self.bettensor_api_client = BettensorAPIClient()
        self.all_games = []
        self.api_hosts = {
            "baseball": "api-baseball.p.rapidapi.com",
            "soccer": "api-football-v1.p.rapidapi.com",
            "nfl": "api.b365api.com"
        }
        self.entropy_system = EntropySystem(max_capacity=256, max_days=45)



    def insert_into_database(self, game_data):
        self.db_manager.execute_query(
            """INSERT INTO game_data (id, team_a, team_b, sport, league, external_id, create_date, last_update_date, event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            game_data
        )

    def update_odds_in_database(self, external_id, team_a_odds, team_b_odds, tie_odds=None):
        if tie_odds is not None:
            self.db_manager.execute_query(
                """UPDATE game_data
                SET team_a_odds = ?, team_b_odds = ?, tie_odds = ?, last_update_date = ?
                WHERE external_id = ?""",
                (team_a_odds, team_b_odds, tie_odds, datetime.now(timezone.utc).isoformat(), external_id)
            )
        else:
            self.db_manager.execute_query(
                """UPDATE game_data
                SET team_a_odds = ?, team_b_odds = ?, last_update_date = ?
                WHERE external_id = ?""",
                (team_a_odds, team_b_odds, datetime.now(timezone.utc).isoformat(), external_id)
            )

    def external_id_exists(self, external_id):
        result = self.db_manager.fetchone(
            """SELECT 1 FROM game_data WHERE external_id = ? LIMIT 1""",
            (external_id,)
        )
        return result is not None

    def get_multiple_game_data(self):
        all_games = []
        sports_config = self.sports_config
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
                            games = self.external_api_client.process_nfl_games()
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
                
                # Set tie_odds to 0.0 for NFL games, otherwise use the value from the game data or 0
                tie_odds = 0.0 if sport.lower() == "nfl" else game['odds'].get('average_tie_odds', 0)
                
                can_tie = sport.lower() == "soccer"

                # Check if the game already exists
                existing_game = self.db_manager.fetchone(
                    "SELECT id FROM game_data WHERE external_id = ?", 
                    (external_id,)
                )

                if existing_game:
                    # Update existing game
                    self.db_manager.execute_query(
                        """
                        UPDATE game_data
                        SET team_a_odds = ?, team_b_odds = ?, tie_odds = ?, last_update_date = ?
                        WHERE external_id = ?
                        """, 
                        (team_a_odds, team_b_odds, tie_odds, last_update_date, external_id)
                    )
                    bt.logging.debug(f"Updated game {external_id} in database")
                else:
                    # Insert new game
                    self.db_manager.execute_query(
                        """
                        INSERT INTO game_data (id, team_a, team_b, sport, league, external_id, create_date, last_update_date,
                                            event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, 
                        (game_id, team_a, team_b, sport, league, external_id, create_date, last_update_date,
                        event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie)
                    )
                    bt.logging.debug(f"Inserted new game {external_id} into database")

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
                games_list = []
                for i in games["response"]:

                    league = i.get('league', {}).get('name', 'unknown_league')

                    game_data = {
                        "home": i["teams"]["home"]["name"],
                        "away": i["teams"]["away"]["name"],
                        "game_id": i["fixture"]["id"] if sport == "soccer" else i["id"],
                        "date": i["fixture"]["date"] if sport == "soccer" else i["date"],
                        "odds": self.get_game_odds(i["fixture"]["id"] if sport == "soccer" else i["id"], sport),
                        "sport": sport,
                        "league": league
                    }
                    
                    if not game_data.get('sport'):
                        bt.logging.error(f"Missing 'sport' field for game {game_data['game_id']}. Setting to default '{sport}'.")
                        game_data['sport'] = sport

                    games_list.append(game_data)
                
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
                        "average_tie_odds": avg_tie_odds if sport == "soccer" else 0.0,
                    }

            # Return None values if no data is available
            return {
                "average_home_odds": 0.0,
                "average_away_odds": 0.0,
                "average_tie_odds": 0.0,
            }
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Error fetching odds for game {game_id} in {sport}: {e}")
            return {
                "average_home_odds": 0.0,
                "average_away_odds": 0.0,
                "average_tie_odds": 0.0,
            }
        



    def get_odds(self, event_id):
        url = f"https://api.b365api.com/v3/bet365/prematch?token={self.bet365_api_key}&FI={event_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            bt.logging.error(f"Failed to fetch odds for event {event_id}")
            return None