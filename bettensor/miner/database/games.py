import datetime
from typing import Dict, Tuple
from bettensor.protocol import TeamGame
import bittensor as bt
from datetime import timedelta

class GamesHandler:
    def __init__(self, db_manager):
        bt.logging.trace("Initializing GamesHandler")
        self.db_manager = db_manager
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
        """
        Add a new game to the database or update an existing one.

        Args:
            cursor: The database cursor.
            game_data (TeamGame): The game data to add or update.
            current_time (datetime.datetime): The current time.

        Returns:
            bool: True if a new game was added, False if an existing game was updated.

        Behavior:
            - Checks if the game already exists in the database.
            - If it exists, updates the game data.
            - If it doesn't exist, inserts a new game record.
        """
        #bt.logging.trace(f"Adding or updating game: {game_data.externalId}")
        event_start_date = datetime.datetime.fromisoformat(game_data.eventStartDate.replace('Z', '+00:00'))
        is_active = 1 if current_time <= event_start_date else 0

        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games WHERE externalID = ?", (game_data.externalId,))
            existing_game = cursor.fetchone()

            if existing_game:
                existing_outcome = existing_game[12]  # Assuming 'outcome' is at index 12
                if existing_outcome != game_data.outcome:
                    bt.logging.debug(f"Game {game_data.externalId} outcome changed from {existing_outcome} to {game_data.outcome}")
                
                cursor.execute(
                    """UPDATE games SET 
                    teamA = ?, teamB = ?, sport = ?, league = ?, 
                    createDate = ?, lastUpdateDate = ?, eventStartDate = ?, active = ?, 
                    outcome = ?, canTie = ? 
                    WHERE externalID = ?""",
                    (
                        game_data.teamA, game_data.teamB,
                        game_data.sport, game_data.league, game_data.createDate, game_data.lastUpdateDate,
                        game_data.eventStartDate, is_active, game_data.outcome,
                        game_data.canTie, game_data.externalId
                    )
                )
                bt.logging.debug(f"Game updated: {game_data.externalId}, New outcome: {game_data.outcome}")
                return False
            else:
                cursor.execute(
                    """INSERT INTO games (
                    gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID, 
                    createDate, lastUpdateDate, eventStartDate, active, outcome, tieOdds, canTie
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        game_data.id, game_data.teamA, game_data.teamAodds, game_data.teamB, game_data.teamBodds,
                        game_data.sport, game_data.league, game_data.externalId, game_data.createDate,
                        game_data.lastUpdateDate, game_data.eventStartDate, is_active,
                        game_data.outcome, game_data.tieOdds, game_data.canTie
                    )
                )
                bt.logging.debug(f"New game added: {game_data.externalId}, Outcome: {game_data.outcome}")
                return True

    def _mark_old_games_inactive(self, current_time: datetime.datetime):
        bt.logging.trace("Marking old games as inactive")
        cutoff_date = current_time - self.inactive_game_window
        with self.db_manager.get_cursor() as cursor:
            cursor.execute(
                "UPDATE games SET active = 0 WHERE eventStartDate < ?",
                (cutoff_date.isoformat(),)
            )

    def _remove_duplicate_games(self):
        bt.logging.trace("Removing duplicate games")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM games
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM games
                    GROUP BY externalID
                )
            """)

    def get_game_by_id(self, game_id: str) -> TeamGame:
        bt.logging.trace(f"Retrieving game with ID: {game_id}")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games WHERE gameID = ?", (game_id,))
            game = cursor.fetchone()
            if game:
                return TeamGame(*game)
        return None

    def get_all_games(self) -> Dict[str, TeamGame]:
        """
        Retrieve all games from the database.

        Returns:
            Dict[str, TeamGame]: A dictionary of all games, keyed by game ID.

        Behavior:
            - Queries the database for all games.
            - Constructs TeamGame objects from the database records.
            - Returns a dictionary of these TeamGame objects.
        """
        bt.logging.trace("Retrieving all games")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games")
            games_raw = cursor.fetchall()

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

    def get_active_games(self) -> Dict[str, TeamGame]:
        """
        Retrieve active games from the database.

        Returns:
            Dict[str, TeamGame]: A dictionary of active games, keyed by game ID.

        Behavior:
            - Queries the database for active games.
            - Constructs TeamGame objects from the database records.
            - Returns a dictionary of these TeamGame objects.
        """
        bt.logging.trace("Retrieving active games")
        current_time = datetime.datetime.now(datetime.timezone.utc)
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM games 
                WHERE eventStartDate > ? 
                ORDER BY eventStartDate ASC
            """, (current_time.isoformat(),))
            games = cursor.fetchall()

        active_games = {}
        for game in games:
            team_game = TeamGame(
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
                canTie=bool(game[14])
            )
            active_games[game[0]] = team_game
        bt.logging.trace(f"Retrieved {len(active_games)} active games")
        return active_games