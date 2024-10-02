import datetime
import traceback
from typing import Dict, Tuple, List, Union, Optional
from bettensor.protocol import TeamGame, TeamGamePrediction
import bittensor as bt
from datetime import datetime, timezone, timedelta
from bettensor.miner.database.predictions import PredictionsHandler
import json
import uuid


class GamesHandler:
    def __init__(self, db_manager, predictions_handler):
        bt.logging.trace("Initializing GamesHandler")
        self.db_manager = db_manager
        self.predictions_handler = predictions_handler
        self.inactive_game_window = timedelta(days=30)
        bt.logging.trace("GamesHandler initialization complete")

    def process_games(self, changed_games):
        bt.logging.trace(f"Processing {len(changed_games)} changed games")
        updated_games = {}
        new_games = {}
        current_time = datetime.now(timezone.utc)

        try:
            self._batch_add_or_update_games(
                changed_games, current_time, updated_games, new_games
            )
            bt.logging.trace(
                f"Processed games: {len(updated_games)} updated, {len(new_games)} new"
            )
        except Exception as e:
            bt.logging.error(f"Error processing games: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

        return updated_games, new_games

    def _batch_add_or_update_games(
        self, game_data_dict, current_time, updated_games, new_games
    ):
        bt.logging.trace(f"Batch adding or updating {len(game_data_dict)} games")
        game_data_dict_by_id = {
            game.game_id: game for game in game_data_dict.values()
        }

        upsert_query = """
        INSERT INTO games (
            game_id, team_a, team_a_odds, team_b, team_b_odds, sport, league,
            create_date, last_update_date, event_start_date, active, outcome, tie_odds, can_tie
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (game_id) DO UPDATE SET
            game_id = EXCLUDED.game_id,
            team_a = EXCLUDED.team_a,
            team_a_odds = EXCLUDED.team_a_odds,
            team_b = EXCLUDED.team_b,
            team_b_odds = EXCLUDED.team_b_odds,
            sport = EXCLUDED.sport,
            league = EXCLUDED.league,
            last_update_date = EXCLUDED.last_update_date,
            event_start_date = EXCLUDED.event_start_date,
            active = EXCLUDED.active::integer,
            outcome = EXCLUDED.outcome,
            tie_odds = EXCLUDED.tie_odds,
            can_tie = EXCLUDED.can_tie
        RETURNING game_id, (xmax = 0) AS is_new, outcome
        """

        params_list = []
        for game_data in game_data_dict_by_id.values():
            event_start_date = self._ensure_timezone_aware(game_data.event_start_date)
            params = (
                game_data.game_id,
                game_data.team_a,
                game_data.team_a_odds,
                game_data.team_b,
                game_data.team_b_odds,
                game_data.sport,
                game_data.league,
                self._ensure_timezone_aware(game_data.create_date),
                self._ensure_timezone_aware(game_data.last_update_date),
                event_start_date,
                int(game_data.active),
                game_data.outcome,
                game_data.tie_odds,
                game_data.can_tie,
            )
            params_list.append(params)

        try:
            self.db_manager.execute_batch(upsert_query, params_list)

            # Fetch the results of the upsert operation
            select_query = "SELECT game_id, (xmax = 0) AS is_new, outcome FROM games WHERE game_id = ANY(%s)"
            results = self.db_manager.execute_query(
                select_query,
                ([game.game_id for game in game_data_dict_by_id.values()],),
            )

            for result in results:
                game_id, is_new, outcome = (
                    result["game_id"],
                    result["is_new"],
                    result["outcome"],
                )
                game_data = game_data_dict_by_id[game_id]
                if is_new:
                    new_games[game_id] = game_data

                else:
                    updated_games[game_id] = game_data


                if outcome != game_data.outcome:
                    bt.logging.info(
                        f"Game {game_id} outcome changed from {outcome} to {game_data.outcome}"
                    )
                    self.predictions_handler.process_game_results(
                        {game_id: game_data}
                    )

            bt.logging.info(
                f"Processed {len(results)} games: {len(new_games)} new, {len(updated_games)} updated"
            )
        except Exception as e:
            bt.logging.error(f"Error in batch add/update games: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _ensure_timezone_aware(self, dt):
        # bt.logging.trace(f"Ensuring timezone awareness for datetime: {dt}")
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def get_game(self, game_id: str) -> Optional[TeamGame]:
        bt.logging.trace(f"Retrieving game with game ID: {game_id}")
        query = "SELECT * FROM games WHERE game_id = %s"
        result = self.db_manager.execute_query(query, (game_id,))
        if result:
            bt.logging.trace(f"Game found: {result[0]}")
            return TeamGame(**result[0])
        bt.logging.trace(f"No game found for game ID: {game_id}")
        return None

    def get_active_games(self):
        bt.logging.trace("Retrieving active games")
        query = """
        SELECT * FROM games
        WHERE active = 1 AND CAST(event_start_date AS TIMESTAMP WITH TIME ZONE) > NOW()
        ORDER BY CAST(event_start_date AS TIMESTAMP WITH TIME ZONE) ASC
        """
        results = self.db_manager.execute_query(query)
        active_games = {}
        for row in results:
            team_game = TeamGame(
                game_id=row["game_id"],
                team_a=row["team_a"],
                team_a_odds=float(row["team_a_odds"]),
                team_b=row["team_b"],
                team_b_odds=float(row["team_b_odds"]),
                sport=row["sport"],
                league=row["league"],
                create_date=self._ensure_iso_format(row["create_date"]),
                last_update_date=self._ensure_iso_format(row["last_update_date"]),
                event_start_date=self._ensure_iso_format(row["event_start_date"]),
                active=bool(row["active"]),
                outcome=row["outcome"],
                tie_odds=float(row["tie_odds"]) if row["tie_odds"] is not None else None,
                can_tie=bool(row["can_tie"]),
            )
            active_games[row["game_id"]] = team_game
        bt.logging.trace(f"Retrieved {len(active_games)} active games")
        return active_games

    def _ensure_iso_format(self, dt):
        if isinstance(dt, datetime):
            return dt.isoformat()
        elif isinstance(dt, str):
            return dt
        else:
            return str(dt)

    def game_exists(self, game_id):
        if not game_id:
            bt.logging.warning(
                "Attempted to check game existence with None or empty game_id"
            )
            return False

        # Ensure game_id is a string
        game_id = str(game_id)

        query = "SELECT COUNT(*) FROM games WHERE game_id = %s"
        bt.logging.info(f"Executing query: {query} with game_id: {game_id}")
        try:
            result = self.db_manager.execute_query(query, (game_id,))
            bt.logging.info(f"Query result: {result}")

            if result and isinstance(result, list) and len(result) > 0:
                count = (
                    result[0]["count"] if isinstance(result[0], dict) else result[0][0]
                )
                exists = int(count) > 0
            else:
                bt.logging.warning(f"Unexpected result format: {result}")
                exists = False

            bt.logging.info(f"Game exists: {exists}")
            return exists
        except Exception as e:
            bt.logging.error(f"Error checking game existence: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False

    def _mark_old_games_inactive(self):
        bt.logging.trace("Marking old games as inactive")
        current_time = datetime.now(timezone.utc)
        cutoff_date = current_time - self.inactive_game_window
        query = "UPDATE games SET active = FALSE WHERE event_start_date < %s"
        self.db_manager.execute_query(query, params=(cutoff_date.isoformat(),))
        bt.logging.trace("Old games marked as inactive")

    def get_upcoming_game_ids(self):
        query = """
        SELECT game_id 
        FROM games 
        WHERE event_start_date > %s 
        ORDER BY event_start_date ASC
        """
        current_time = datetime.now(timezone.utc)
        try:
            result = self.db_manager.execute_query(query, (current_time,))
            return [row["game_id"] for row in result] if result else []
        except Exception as e:
            bt.logging.error(f"Error fetching upcoming game IDs: {str(e)}")
            return []

    def request_upcoming_game_ids(self, redis_client):
        message_id = str(uuid.uuid4())
        message = {"action": "get_upcoming_game_ids", "message_id": message_id}
        redis_client.publish("game_requests", json.dumps(message))
        return message_id

    def get_games_by_sport(self, sport: str) -> Dict[str, TeamGame]:
        bt.logging.warning(f"Retrieving games by sport: {sport}")
        query = """
        SELECT game_id, team_a, team_a_odds, team_b, team_b_odds, sport, league, create_date, last_update_date, event_start_date, active, outcome, tie_odds, can_tie
        FROM games
        WHERE active = 1 AND LOWER(sport) = LOWER(%s)
        """
        results = self.db_manager.execute_query(query, (sport,))
        games = {}
        for row in results:
            team_game = TeamGame(
                game_id=row["game_id"],
                team_a=row["team_a"],
                team_b=row["team_b"],
                sport=row["sport"],
                league=row["league"],
                create_date=self._ensure_iso_format(row["create_date"]),
                last_update_date=self._ensure_iso_format(row["last_update_date"]),
                event_start_date=self._ensure_iso_format(row["event_start_date"]),
                active=bool(row["active"]),
                outcome=row["outcome"],
                team_a_odds=float(row["team_a_odds"]),
                team_b_odds=float(row["team_b_odds"]),
                tie_odds=float(row["tie_odds"]) if row["tie_odds"] is not None else None,
                can_tie=bool(row["can_tie"]),
            )
            games[row["game_id"]] = team_game
        return games
