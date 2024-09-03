import requests
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from dateutil import parser
import sqlite3
import bittensor as bt
import os


class SportsData:
    def __init__(self, db_name="data/validator.db"):
        self.db_name = db_name
        self.rapid_api_key = os.getenv("RAPID_API_KEY")
        self.api_hosts = {
            "baseball": "api-baseball.p.rapidapi.com",
            "soccer": "api-football-v1.p.rapidapi.com",
        }
        self.create_database()
        self.all_games = []

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
        bt.logging.info("Fetching games from RapidAPI")
        all_games = []
        for sport, leagues in sports_config.items():
            for league_info in leagues:
                league = league_info['id']
                season = league_info.get('season', '2024')
                try:
                    games = self.get_game_data(sport=sport, league=league, season=season)
                    all_games.extend(games)
                except StopIteration:
                    bt.logging.warning(f"StopIteration encountered while fetching data for {sport}, league {league}. Skipping.")
                except Exception as e:
                    bt.logging.error(f"Error fetching data for {sport}, league {league}: {e}")
        return all_games

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
