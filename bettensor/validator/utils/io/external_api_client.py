import os
import json
import time
import requests
import bittensor as bt
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from .sports_config import sports_config
from .base_api_client import BaseAPIClient
from datetime import datetime, timedelta, timezone


class ExternalAPIClient(BaseAPIClient):
    def __init__(self):
        super().__init__()
        self._load_env_variables()
        self.sports_config = sports_config
        self.api_hosts = {
            "baseball": "api-baseball.p.rapidapi.com",
            "soccer": "api-football-v1.p.rapidapi.com",
            "nfl": "api.b365api.com",
        }
        self.last_request_time = 0
        self.min_request_interval = 1

    def _load_env_variables(self):
        try:
            load_dotenv()
            self.rapid_api_key = os.getenv("RAPID_API_KEY")
            self.bet365_api_key = os.getenv("BET365_API_KEY")
        except Exception as e:
            bt.logging.error(f"Failed to load environment variables: {e}")

    def fetch_all_game_data(self):
        all_games = []
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=6)

        for sport, leagues in self.sports_config.items():
            if sport in self.api_hosts:
                for league in leagues:
                    league_id, season = league["id"], league["season"]
                    bt.logging.debug(
                        f"Fetching data for {sport}, league ID: {league_id}, season: {season}"
                    )

                    sport_games = self._fetch_sport_games(
                        sport, league_id, str(season), start_date, end_date
                    )
                    if sport_games:
                        all_games.extend(sport_games)
                    else:
                        bt.logging.warning(
                            f"No games fetched for {sport}, league ID: {league_id}, season: {season}"
                        )

        bt.logging.info(f"Total games fetched: {len(all_games)}")
        filtered_games = self.filter_games(all_games)
        bt.logging.info(
            f"Filtered {len(all_games) - len(filtered_games)} games out of {len(all_games)} total games"
        )
        return filtered_games

    def _fetch_sport_games(self, sport, league, season, start_date, end_date):
        if sport == "nfl":
            return self._fetch_nfl_games()

        url = f"https://{self.api_hosts[sport]}/{'v3/fixtures' if sport == 'soccer' else 'games'}"
        games = []

        date_range = (
            [start_date + timedelta(n) for n in range(7)]
            if sport == "baseball"
            else [start_date]
        )

        for single_date in date_range:
            querystring = {
                "league": league,
                "season": season,
                "date"
                if sport == "baseball"
                else "from": single_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d") if sport == "soccer" else None,
            }
            querystring = {k: v for k, v in querystring.items() if v is not None}
            games.extend(self._fetch_games(url, querystring, sport))

        bt.logging.debug(
            f"Initially fetched {len(games)} games for {sport}, league {league}"
        )
        return games

    def _fetch_games(self, url, querystring, sport):
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": self.api_hosts[sport],
        }

        try:
            bt.logging.debug(f"Sending request to {url} with params: {querystring}")
            response = self._make_request(url, headers, querystring)
            games = response.json()

            if "response" in games:
                return self._process_games(games["response"], sport)
            else:
                bt.logging.warning(f"Unexpected response format: {games}")
                return []
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            bt.logging.error(f"Error fetching games: {e}")
            return []

    def _make_request(self, url, headers, params):
        while True:
            current_time = time.time()
            if current_time - self.last_request_time < self.min_request_interval:
                time.sleep(
                    self.min_request_interval - (current_time - self.last_request_time)
                )

            response = requests.get(url, headers=headers, params=params)
            self.last_request_time = time.time()

            if response.status_code == 429:
                bt.logging.warning(
                    "Rate limit exceeded. Waiting for 60 seconds before retrying."
                )
                time.sleep(60)
            else:
                response.raise_for_status()
                return response

    def _process_games(self, games, sport):
        games_list = []
        for game in games:
            game_data = {
                "home": game["teams"]["home"]["name"],
                "away": game["teams"]["away"]["name"],
                "game_id": game["fixture"]["id"] if sport == "soccer" else game["id"],
                "date": game["fixture"]["date"] if sport == "soccer" else game["date"],
                "odds": self.get_game_odds(
                    game["fixture"]["id"] if sport == "soccer" else game["id"], sport
                ),
                "sport": sport,
                "league": game.get("league", {}).get("name", "unknown_league"),
            }
            games_list.append(game_data)

        bt.logging.debug(f"Processed {len(games_list)} games for {sport}")
        return games_list

    def get_game_odds(self, game_id, sport):
        url = f"https://{self.api_hosts[sport]}/{'v3/odds' if sport == 'soccer' else 'odds'}"
        querystring = {"fixture" if sport == "soccer" else "game": game_id}
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": self.api_hosts[sport],
        }

        try:
            bt.logging.debug(f"Fetching odds for game {game_id} in {sport}")
            response = self._make_request(url, headers, querystring)
            odds_data = response.json()

            return self._process_odds(odds_data, sport)
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Error fetching odds for game {game_id} in {sport}: {e}")
            return self._default_odds()

    def _process_odds(self, odds_data, sport):
        total_home_odds = total_away_odds = total_tie_odds = count = 0

        if odds_data.get("response") and odds_data["response"][0].get("bookmakers"):
            for bookmaker in odds_data["response"][0]["bookmakers"]:
                for bet in bookmaker.get("bets", []):
                    if (
                        sport == "soccer"
                        and bet["name"] == "Match Winner"
                        and len(bet["values"]) >= 3
                    ) or (
                        sport == "baseball"
                        and bet["name"] == "Home/Away"
                        and len(bet["values"]) >= 2
                    ):
                        odds_dict = {
                            odd["value"]: float(odd["odd"]) for odd in bet["values"]
                        }
                        home_odds = odds_dict.get("Home")
                        away_odds = odds_dict.get("Away")
                        tie_odds = odds_dict.get("Draw") if sport == "soccer" else None

                        if (
                            home_odds
                            and away_odds
                            and (tie_odds is not None or sport == "baseball")
                        ):
                            total_home_odds += home_odds
                            total_away_odds += away_odds
                            if sport == "soccer":
                                total_tie_odds += tie_odds
                            count += 1
                        else:
                            bt.logging.warning(
                                f"Missing odds for game in bookmaker {bookmaker['name']}"
                            )

        if count > 0:
            return {
                "average_home_odds": round(total_home_odds / count, 2),
                "average_away_odds": round(total_away_odds / count, 2),
                "average_tie_odds": round(total_tie_odds / count, 2)
                if sport == "soccer"
                else None,
            }

        return self._default_odds()

    def _default_odds(self):
        return {
            "average_home_odds": None,
            "average_away_odds": None,
            "average_tie_odds": None,
        }

    def filter_games(self, games):
        return [
            game
            for game in games
            if (
                game["odds"]["average_home_odds"] is not None
                and game["odds"]["average_away_odds"] is not None
                and game["odds"]["average_home_odds"] >= 1.05
                and game["odds"]["average_away_odds"] >= 1.05
                and not (
                    game["odds"]["average_home_odds"] == 1.5
                    and game["odds"]["average_away_odds"] == 3.0
                    and game["odds"].get("average_tie_odds") == 1.5
                )
            )
        ]

    def _fetch_nfl_games(self):
        api_response = self._get_upcoming_nfl_events()
        if not api_response:
            return []

        nfl_games = []
        for game in api_response.get("results", []):
            if game.get("league", {}).get("name") == "NFL":
                game_data = self._process_nfl_game(game)
                if game_data:
                    nfl_games.append(game_data)

        bt.logging.info(f"Processed {len(nfl_games)} NFL games")
        return nfl_games

    def _get_upcoming_nfl_events(self):
        url = "https://api.b365api.com/v1/bet365/upcoming"
        params = {"sport_id": 12, "token": self.bet365_api_key}
        headers = {}  # No special headers needed for this API
        try:
            response = self._make_request(url, headers, params)
            return response.json()
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Failed to fetch upcoming NFL events: {e}")
            return None

    def _process_nfl_game(self, game):
        game_id = game["id"]
        home_team = game["home"]["name"]
        away_team = game["away"]["name"]
        game_time = datetime.fromtimestamp(
            int(game["time"]), tz=timezone.utc
        ).isoformat()

        bt.logging.debug(f"Fetching odds for {away_team} @ {home_team} ({game_time})")
        odds_data = self._get_nfl_odds(game_id)

        if odds_data:
            moneyline_odds = self._get_moneyline_odds(odds_data, home_team, away_team)
            if moneyline_odds:
                return {
                    "home": home_team,
                    "away": away_team,
                    "game_id": str(game_id),
                    "date": game_time,
                    "odds": {
                        "average_home_odds": float(moneyline_odds[home_team]["odds"]),
                        "average_away_odds": float(moneyline_odds[away_team]["odds"]),
                        "average_tie_odds": None,
                    },
                    "sport": "Football",
                    "league": "NFL",
                }
            else:
                bt.logging.warning(f"No moneyline odds found for game {game_id}")
        else:
            bt.logging.warning(f"Failed to fetch odds for game {game_id}")

        return None

    def _get_nfl_odds(self, event_id: str):
        url = "https://api.b365api.com/v3/bet365/prematch"
        params = {"FI": event_id, "token": self.bet365_api_key}
        headers = {}  # No special headers needed for this API
        try:
            response = self._make_request(url, headers, params)
            return response.json()
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Failed to fetch NFL odds for event {event_id}: {e}")
            return None

    def _get_moneyline_odds(
        self, odds_data: Dict[str, Any], home_team: str, away_team: str
    ) -> Dict[str, Dict[str, Any]]:
        moneyline_odds = {}
        for market in (
            odds_data.get("results", [{}])[0].get("main", {}).get("sp", {}).values()
        ):
            if market.get("name") == "Game Lines":
                for odd in market.get("odds", []):
                    if odd.get("name") == "Money Line":
                        team = home_team if odd["header"] == "1" else away_team
                        moneyline_odds[team] = {
                            "odds": odd["odds"],
                            "implied_probability": f"{(1 / float(odd['odds'])) * 100:.2f}%",
                        }
        return moneyline_odds
