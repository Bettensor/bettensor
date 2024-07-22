import datetime
from typing import Dict, Tuple
from bettensor.protocol import TeamGame
from bettensor.miner.database.database_manager import DatabaseManager
import bittensor as bt

class GamesHandler:
    def __init__(self, db_manager: DatabaseManager, history_days: int = 30):
        """
        Initialize the GamesHandler.

        Args:
            db_manager (DatabaseManager): The database manager instance.
            history_days (int): The number of days to keep game history.

        Behavior:
            - Sets up the database manager and history days for handling games.
        """
        bt.logging.trace(f"Initializing GamesHandler with history_days: {history_days}")
        self.db_manager = db_manager
        self.history_days = history_days
        bt.logging.trace("GamesHandler initialization complete")

    def process_games(self, game_data_dict: Dict[str, TeamGame]) -> Tuple[Dict[str, TeamGame], Dict[str, TeamGame]]:
        """
        Process a dictionary of games, updating existing games and adding new ones.

        Args:
            game_data_dict (Dict[str, TeamGame]): A dictionary of games to process.

        Returns:
            Tuple[Dict[str, TeamGame], Dict[str, TeamGame]]: A tuple containing two dictionaries:
                - The first dictionary contains updated games.
                - The second dictionary contains new games.

        Behavior:
            - Updates existing games in the database.
            - Adds new games to the database.
            - Marks old games as inactive.
            - Removes duplicate games.
        """
        bt.logging.trace(f"Processing {len(game_data_dict)} games")
        try:
            current_time = datetime.datetime.now(datetime.timezone.utc)
            cutoff_date = current_time - datetime.timedelta(days=self.history_days)
            updated_games = {}
            new_games = {}
            
            with self.db_manager.get_cursor() as cursor:
                for game_id, game_data in game_data_dict.items():
                    is_new = self._add_or_update_game(cursor, game_data, current_time)
                    if is_new:
                        new_games[game_id] = game_data
                    else:
                        updated_games[game_id] = game_data

                # Mark old games as inactive
                cursor.execute(
                    "UPDATE games SET active = 0 WHERE eventStartDate < ?",
                    (cutoff_date.isoformat(),)
                )

                # Remove duplicate games
                cursor.execute("""
                    DELETE FROM games
                    WHERE rowid NOT IN (
                        SELECT MIN(rowid)
                        FROM games
                        GROUP BY externalID
                    )
                """)

            bt.logging.trace(f"Game processing complete. Updated games: {len(updated_games)}, New games: {len(new_games)}")
            return updated_games, new_games
        except Exception as e:
            bt.logging.error(f"Error processing games: {e}")
            return {}, {}

    def _add_or_update_game(self, cursor, game_data: TeamGame, current_time: datetime.datetime) -> bool:
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
        bt.logging.trace(f"Adding or updating game: {game_data.externalId}")
        event_start_date = datetime.datetime.fromisoformat(game_data.eventStartDate.replace('Z', '+00:00'))
        is_active = 1 if current_time <= event_start_date else 0

        cursor.execute("SELECT * FROM games WHERE externalID = ?", (game_data.externalId,))
        existing_game = cursor.fetchone()

        if existing_game:
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
            bt.logging.trace(f"Game updated: {game_data.externalId}")
            return False
        else:
            cursor.execute(
                """INSERT INTO games (
                gameID, teamA, teamAodds, teamB, teamBodds, sport, league, externalID, 
                createDate, lastUpdateDate, eventStartDate, active, outcome, tieOdds, canTie
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    game_data.id, game_data.teamA, game_data.teamAodds, game_data.teamB, game_data.teamBodds,
                    game_data.sport, game_data.league, game_data.externalId, game_data.createDate,
                    game_data.lastUpdateDate, game_data.eventStartDate, is_active,
                    game_data.outcome, game_data.tieOdds, game_data.canTie
                )
            )
            bt.logging.trace(f"Game added: {game_data.externalId}")
            return True

    def get_active_games(self) -> Dict[str, TeamGame]:
        """
        Retrieve all active games from the database.

        Returns:
            Dict[str, TeamGame]: A dictionary of active games, keyed by game ID.

        Behavior:
            - Queries the database for all games marked as active.
            - Constructs TeamGame objects from the database records.
            - Returns a dictionary of these TeamGame objects.
        """
        bt.logging.trace("Retrieving active games")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games WHERE active = 1")
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
        bt.logging.trace(f"Retrieved {len(game_dict)} active games")
        return game_dict