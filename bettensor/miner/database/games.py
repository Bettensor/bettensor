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
        game_data_dict_by_external_id = {
            game.externalId: game for game in game_data_dict.values()
        }

        upsert_query = """
        INSERT INTO games (
            gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID, 
            createDate, lastUpdateDate, eventStartDate, active, outcome, tieOdds, canTie
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (externalID) DO UPDATE SET
            gameID = EXCLUDED.gameID,
            teamA = EXCLUDED.teamA,
            teamAodds = EXCLUDED.teamAodds,
            teamB = EXCLUDED.teamB,
            teamBodds = EXCLUDED.teamBodds,
            sport = EXCLUDED.sport,
            league = EXCLUDED.league,
            lastUpdateDate = EXCLUDED.lastUpdateDate,
            eventStartDate = EXCLUDED.eventStartDate,
            active = EXCLUDED.active,
            outcome = EXCLUDED.outcome,
            tieOdds = EXCLUDED.tieOdds,
            canTie = EXCLUDED.canTie
        RETURNING externalID, (xmax = 0) AS is_new, outcome
        """

        params_list = []
        for game_data in game_data_dict_by_external_id.values():
            event_start_date = self._ensure_timezone_aware(game_data.eventStartDate)
            is_active = (
                1 if current_time <= datetime.fromisoformat(event_start_date) else 0
            )
            params = (
                game_data.id,
                game_data.teamA,
                game_data.teamAodds,
                game_data.teamB,
                game_data.teamBodds,
                game_data.sport,
                game_data.league,
                game_data.externalId,
                self._ensure_timezone_aware(game_data.createDate),
                self._ensure_timezone_aware(game_data.lastUpdateDate),
                event_start_date,
                is_active,
                game_data.outcome,
                game_data.tieOdds,
                game_data.canTie,
            )
            params_list.append(params)

        try:
            self.db_manager.execute_batch(upsert_query, params_list)

            # Fetch the results of the upsert operation
            select_query = "SELECT externalID, (xmax = 0) AS is_new, outcome FROM games WHERE externalID = ANY(%s)"
            results = self.db_manager.execute_query(
                select_query,
                ([game.externalId for game in game_data_dict_by_external_id.values()],),
            )

            for result in results:
                external_id, is_new, outcome = (
                    result["externalid"],
                    result["is_new"],
                    result["outcome"],
                )
                game_data = game_data_dict_by_external_id[external_id]
                if is_new:
                    new_games[external_id] = game_data
                    # bt.logging.debug(f"New game added: {external_id}")
                else:
                    updated_games[external_id] = game_data
                    # bt.logging.debug(f"Game updated: {external_id}")

                if outcome != game_data.outcome:
                    bt.logging.info(
                        f"Game {external_id} outcome changed from {outcome} to {game_data.outcome}"
                    )
                    self.predictions_handler.process_game_results(
                        {external_id: game_data}
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

    def get_game(self, external_id: str) -> Optional[TeamGame]:
        bt.logging.trace(f"Retrieving game with external ID: {external_id}")
        query = "SELECT * FROM games WHERE externalID = %s"
        result = self.db_manager.execute_query(query, (external_id,))
        if result:
            bt.logging.trace(f"Game found: {result[0]}")
            return TeamGame(**result[0])
        bt.logging.trace(f"No game found for external ID: {external_id}")
        return None

    def get_active_games(self):
        bt.logging.trace("Retrieving active games")
        query = """
        SELECT * FROM games
        WHERE active = 1 AND CAST(eventStartDate AS TIMESTAMP WITH TIME ZONE) > NOW()
        ORDER BY CAST(eventStartDate AS TIMESTAMP WITH TIME ZONE) ASC
        """
        results = self.db_manager.execute_query(query)
        active_games = {}
        for row in results:
            team_game = TeamGame(
                id=row["gameid"],
                teamA=row["teama"],
                teamAodds=float(row["teamaodds"]),
                teamB=row["teamb"],
                teamBodds=float(row["teambodds"]),
                sport=row["sport"],
                league=row["league"],
                externalId=row["externalid"],
                createDate=self._ensure_iso_format(row["createdate"]),
                lastUpdateDate=self._ensure_iso_format(row["lastupdatedate"]),
                eventStartDate=self._ensure_iso_format(row["eventstartdate"]),
                active=bool(row["active"]),
                outcome=row["outcome"],
                tieOdds=float(row["tieodds"]) if row["tieodds"] is not None else None,
                canTie=bool(row["cantie"]),
            )
            active_games[row["externalid"]] = team_game
        bt.logging.trace(f"Retrieved {len(active_games)} active games")
        return active_games

    def _ensure_iso_format(self, dt):
        if isinstance(dt, datetime):
            return dt.isoformat()
        elif isinstance(dt, str):
            return dt
        else:
            return str(dt)

    def game_exists(self, external_id):
        if not external_id:
            bt.logging.warning(
                "Attempted to check game existence with None or empty external_id"
            )
            return False

        # Ensure external_id is a string
        external_id = str(external_id)

        query = "SELECT COUNT(*) FROM games WHERE externalID = %s"
        bt.logging.info(f"Executing query: {query} with external_id: {external_id}")
        try:
            result = self.db_manager.execute_query(query, (external_id,))
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
        query = "UPDATE games SET active = FALSE WHERE eventStartDate < %s"
        self.db_manager.execute_query(query, params=(cutoff_date.isoformat(),))
        bt.logging.trace("Old games marked as inactive")

    def get_upcoming_game_ids(self):
        query = """
        SELECT externalID 
        FROM games 
        WHERE eventStartDate > %s 
        ORDER BY eventStartDate ASC
        """
        current_time = datetime.now(timezone.utc)
        try:
            result = self.db_manager.execute_query(query, (current_time,))
            return [row["externalid"] for row in result] if result else []
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
        SELECT gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID, createDate, lastUpdateDate, eventStartDate, active, outcome, tieOdds, canTie
        FROM games
        WHERE active = 1 AND LOWER(sport) = LOWER(%s)
        """
        results = self.db_manager.execute_query(query, (sport,))
        games = {}
        for row in results:
            team_game = TeamGame(
                id=row["gameid"],
                teamA=row["teama"],
                teamB=row["teamb"],
                sport=row["sport"],
                league=row["league"],
                externalId=row["externalid"],
                createDate=self._ensure_iso_format(row["createdate"]),
                lastUpdateDate=self._ensure_iso_format(row["lastupdatedate"]),
                eventStartDate=self._ensure_iso_format(row["eventstartdate"]),
                active=bool(row["active"]),
                outcome=row["outcome"],
                teamAodds=float(row["teamaodds"]),
                teamBodds=float(row["teambodds"]),
                tieOdds=float(row["tieodds"]) if row["tieodds"] is not None else None,
                canTie=bool(row["cantie"]),
            )
            games[row["externalid"]] = team_game
        return games
