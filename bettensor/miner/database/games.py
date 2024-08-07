import datetime
from typing import Dict, Tuple
from bettensor.protocol import TeamGame
import bittensor as bt
from datetime import timedelta
from bettensor.miner.database.predictions import PredictionsHandler

class GamesHandler:
    def __init__(self, db_manager, predictions_handler):
        bt.logging.trace("Initializing GamesHandler")
        self.db_manager = db_manager
        self.predictions_handler = predictions_handler
        self.inactive_game_window = timedelta(days=30)
        bt.logging.trace("GamesHandler initialization complete")

    def process_games(self, game_data_dict: Dict[str, TeamGame]) -> Dict[str, TeamGame]:
        bt.logging.info(f"Processing {len(game_data_dict)} games")
        try:
            current_time = datetime.datetime.now(datetime.timezone.utc)
            updated_games = {}
            new_games = {}
            
            for game_id, game_data in game_data_dict.items():
                is_new = self._add_or_update_game(game_data, current_time)
                if is_new:
                    new_games[game_id] = game_data
                else:
                    updated_games[game_id] = game_data

            self._mark_old_games_inactive(current_time)
            self._remove_duplicate_games()

            bt.logging.info(f"Game processing complete. Updated: {len(updated_games)}, New: {len(new_games)}")
            return updated_games, new_games
        except Exception as e:
            bt.logging.error(f"Error processing games: {e}")
            return {}, {}

    def _add_or_update_game(self, game_data: TeamGame, current_time: datetime.datetime) -> bool:
        bt.logging.debug(f"Adding or updating game: {game_data.externalId}")
        event_start_date = datetime.datetime.fromisoformat(game_data.eventStartDate.replace('Z', '+00:00'))
        is_active = bool(current_time <= event_start_date)

        query = "SELECT * FROM games WHERE externalID = %s::text"
        existing_game = self.db_manager.execute_query(query, params=(game_data.externalId,))

        if existing_game:
            existing_outcome = existing_game[0][12]  # Assuming 'outcome' is at index 12
            bt.logging.debug(f"Existing game found: {game_data.externalId}, Current outcome: {existing_outcome}, New outcome: {game_data.outcome}")
            
            if existing_outcome != game_data.outcome:
                bt.logging.info(f"Game {game_data.externalId} outcome changed from {existing_outcome} to {game_data.outcome}")
            
            query = """UPDATE games SET 
                teamA = %s, teamAodds = %s, teamB = %s, teamBodds = %s, sport = %s, league = %s, 
                createDate = %s, lastUpdateDate = %s, eventStartDate = %s, active = %s::boolean, 
                outcome = %s, tieOdds = %s, canTie = %s::boolean 
                WHERE externalID = %s::text"""
            params = (
                game_data.teamA, game_data.teamAodds, game_data.teamB, game_data.teamBodds,
                game_data.sport, game_data.league, game_data.createDate, game_data.lastUpdateDate,
                game_data.eventStartDate, is_active, game_data.outcome, game_data.tieOdds,
                game_data.canTie, game_data.externalId
            )
            self.db_manager.execute_query(query, params=params)
            bt.logging.debug(f"Game updated: {game_data.externalId}, New outcome: {game_data.outcome}")
            
            # Trigger prediction update
            bt.logging.debug(f"Triggering prediction update for game: {game_data.externalId}")
            self.predictions_handler.process_game_results({game_data.id: game_data})
            
            return False
        else:
            bt.logging.debug(f"Adding new game: {game_data.externalId}")
            query = """INSERT INTO games (
                gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID, 
                createDate, lastUpdateDate, eventStartDate, active, outcome, tieOdds, canTie
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::text, %s, %s, %s, %s::boolean, %s, %s, %s::boolean)"""
            params = (
                game_data.id, game_data.teamA, game_data.teamAodds, game_data.teamB, game_data.teamBodds,
                game_data.sport, game_data.league, game_data.externalId, game_data.createDate,
                game_data.lastUpdateDate, game_data.eventStartDate, is_active,
                game_data.outcome, game_data.tieOdds, game_data.canTie
            )
            self.db_manager.execute_query(query, params=params)
            bt.logging.debug(f"New game added: {game_data.externalId}, Outcome: {game_data.outcome}")
            
            # Trigger prediction update
            bt.logging.debug(f"Triggering prediction update for new game: {game_data.externalId}")
            self.predictions_handler.process_game_results({game_data.id: game_data})
            
            return True

    def _mark_old_games_inactive(self, current_time: datetime.datetime):
        bt.logging.trace("Marking old games as inactive")
        cutoff_date = current_time - self.inactive_game_window
        query = "UPDATE games SET active = FALSE WHERE eventStartDate < %s"
        self.db_manager.execute_query(query, params=(cutoff_date.isoformat(),))

    def _remove_duplicate_games(self):
        bt.logging.trace("Removing duplicate games")
        query = """
            DELETE FROM games
            WHERE ctid NOT IN (
                SELECT MIN(ctid)
                FROM games
                GROUP BY externalID
            )
        """
        self.db_manager.execute_query(query)

    def get_game_by_id(self, game_id: str) -> TeamGame:
        bt.logging.trace(f"Retrieving game with ID: {game_id}")
        query = "SELECT * FROM games WHERE gameID = %s"
        result = self.db_manager.execute_query(query, params=(game_id,))
        if result and result[0]:
            return TeamGame(*result[0])
        return None

    def get_all_games(self) -> Dict[str, TeamGame]:
        bt.logging.trace("Retrieving all games")
        query = "SELECT * FROM games"
        games_raw = self.db_manager.execute_query(query)

        game_dict = {}
        for game in games_raw:
            single_game = TeamGame(
                id=game[0],
                teamA=game[1],
                teamAodds=game[2],
                teamB=game[3],
                teamBodds=game[4],
                sport=game[5],
                league=game[6],
                externalId=game[7],
                createDate=game[8],
                lastUpdateDate=game[9],
                eventStartDate=game[10],
                active=bool(game[11]),
                outcome=game[12],
                tieOdds=game[13],
                canTie=bool(game[14]),
            )
            game_dict[game[0]] = single_game
        bt.logging.trace(f"Retrieved {len(game_dict)} games")
        return game_dict

    def get_active_games(self):
        bt.logging.trace("Retrieving active games")
        current_time = datetime.datetime.now(datetime.timezone.utc)
        query = """
        SELECT * FROM games
        WHERE active = true AND eventstartdate > NOW()
        ORDER BY eventstartdate ASC
        """
        results = self.db_manager.execute_query(query)
        active_games = {}
        for row in results:
            # Convert datetime objects to ISO format strings
            create_date = row['createdate'].isoformat() if isinstance(row['createdate'], datetime.datetime) else row['createdate']
            last_update_date = row['lastupdatedate'].isoformat() if isinstance(row['lastupdatedate'], datetime.datetime) else row['lastupdatedate']
            event_start_date = row['eventstartdate'].isoformat() if isinstance(row['eventstartdate'], datetime.datetime) else row['eventstartdate']

            team_game = TeamGame(
                id=row['gameid'],
                teamA=row['teama'],
                teamAodds=float(row['teamaodds']),
                teamB=row['teamb'],
                teamBodds=float(row['teambodds']),
                sport=row['sport'],
                league=row['league'],
                externalId=row['externalid'],
                createDate=create_date,
                lastUpdateDate=last_update_date,
                eventStartDate=event_start_date,
                active=bool(row['active']),
                outcome=row['outcome'],
                tieOdds=float(row['tieodds']),
                canTie=bool(row['cantie'])
            )
            active_games[row['gameid']] = team_game
        bt.logging.trace(f"Retrieved {len(active_games)} active games")
        return active_games

    def game_exists(self, game_id):
        query = "SELECT COUNT(*) FROM games WHERE externalID = %s"
        result = self.db_manager.execute_query(query, (game_id,))
        return result[0][0] > 0