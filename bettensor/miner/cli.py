import argparse
import datetime
import json
import signal
import sqlite3
import traceback
import uuid
import pytz
import subprocess
import bittensor as bt
import rich
import prompt_toolkit
from rich.console import Console
from rich.table import Table
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import Frame, TextArea, Label
from prompt_toolkit.styles import Style
from prompt_toolkit.application import Application as PTApplication
from argparse import ArgumentParser
import time
import threading
from prompt_toolkit.layout.containers import Window, HSplit
import logging
import os
from bettensor.miner.database.database_manager import get_db_manager

log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename="./logs/cli_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)


global_style = Style.from_dict(
    {
        "text-area": "fg:green",
        "frame": "fg:green",
        "label": "fg:green",
        "wager-input": "fg:green",
    }
)


class Application:
    def __init__(self):
        """
        Initialize the CLI Application.

        Behavior:
            - Sets up the database connection
            - Retrieves available miners
            - Selects a miner to work with
            - Loads miner data and initializes the UI
        """
        self.db_manager = get_db_manager()
        
        if not os.path.exists(self.db_manager.db_path):
            print("Error: Database not found. Please start the miner first.")
            exit(1)

        self.available_miners = self.get_available_miners()
        
        if not self.available_miners:
            print("Error: No miners found in the database. Please start the miner first.")
            exit(1)

        self.miner_uid = self.select_miner()
        self.load_miner_data()

        print(
            f"Miner UID: {self.miner_uid}, Miner Hotkey: {self.miner_hotkey}, DB Path: {self.db_manager.db_path}"
        )

        self.reload_data()
        self.current_view = MainMenu(self)
        root_container = self.current_view.box
        self.layout = Layout(root_container)
        self.style = global_style
        self.check_db_init()

    def get_available_miners(self):
        """
        Retrieve all available miners from the database.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing miner UIDs and hotkeys.

        Behavior:
            - Queries the database for all miner UIDs and hotkeys
        """
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT miner_uid, miner_hotkey FROM miner_stats")
            return cursor.fetchall()

    def select_miner(self):
        """
        Prompt the user to select a miner if multiple miners are available.

        Returns:
            str: The selected miner's UID.

        Behavior:
            - If only one miner is available, it's automatically selected
            - If multiple miners are available, the user is prompted to choose one
        """
        if len(self.available_miners) == 1:
            print(f"Only one miner found. Selecting UID: {self.available_miners[0][0]}")
            return self.available_miners[0][0]

        print("Available miners:")
        for i, (uid, hotkey) in enumerate(self.available_miners, 1):
            print(f"{i}. UID: {uid}, Hotkey: {hotkey}")

        while True:
            try:
                choice = int(input("Enter the number of the miner you want to select: "))
                if 1 <= choice <= len(self.available_miners):
                    return self.available_miners[choice-1][0]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def load_miner_data(self):
        """
        Load data for the selected miner.

        Behavior:
            - Retrieves the miner's hotkey from the database
            - Sets the miner_hotkey attribute
        """
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT miner_hotkey FROM miner_stats WHERE miner_uid = ?", (self.miner_uid,))
            result = cursor.fetchone()
            if result:
                self.miner_hotkey = result[0]
            else:
                print(f"No miner found with UID: {self.miner_uid}")
                exit(1)

    def reload_data(self):
        """
        Reload all data for the current miner.

        Behavior:
            - Retrieves predictions, game data, and miner stats from the database
            - Initializes default values if no stats are found
            - Sets up active games and unsubmitted predictions
        """
        self.predictions = self.get_predictions()
        self.games = self.get_game_data()
        self.miner_stats = self.get_miner_stats(self.miner_uid)
        
        if self.miner_stats is None:
            print(f"No miner stats found for UID: {self.miner_uid}. Initializing with default values.")
            self.miner_stats = {
                "miner_uid": self.miner_uid,
                "miner_hotkey": self.miner_hotkey,
                "miner_cash": 1000,  # Default starting cash
                "miner_status": "active",
                "miner_last_prediction_date": None,
                "miner_lifetime_earnings": 0,
                "miner_lifetime_wager": 0,
                "miner_lifetime_predictions": 0,
                "miner_lifetime_wins": 0,
                "miner_lifetime_losses": 0,
                "miner_win_loss_ratio": 0,
                "miner_rank": 0,
                "miner_current_incentive": 0,
            }
            # Insert these default stats into the database
            self.insert_miner_stats(self.miner_stats)

        self.active_games = {}
        self.unsubmitted_predictions = {}
        self.miner_cash = self.miner_stats["miner_cash"]

    def change_view(self, new_view):
        """
        Change the current view of the CLI.

        Args:
            new_view: The new view to display.

        Behavior:
            - Updates the current_view attribute
            - Changes the layout container to the new view
        """
        self.current_view = new_view
        self.layout.container = new_view.box

    def run(self):
        """
        Run the CLI application.

        Behavior:
            - Creates and runs the prompt_toolkit Application
        """
        self.app = PTApplication(
            layout=self.layout,
            key_bindings=bindings,
            full_screen=True,
            style=self.style,
        )
        self.app.custom_app = self
        self.app.run()

    def check_db_init(self):
        """
        Check if the database is properly initialized.

        Behavior:
            - Attempts to query the predictions table
            - If an exception occurs, prints an error message
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM predictions")
        except Exception as e:
            print(e)
            print("Database not initialized properly, restart your miner first")

    def submit_predictions(self):
        """
        Submit unsubmitted predictions to the database.

        Behavior:
            - Iterates through unsubmitted predictions
            - Inserts or replaces predictions in the database
            - Logs warnings for any insertion failures
        """
        with self.db_manager.get_cursor() as cursor:
            for prediction in self.unsubmitted_predictions.values():
                try:
                    cursor.execute(
                        """INSERT OR REPLACE INTO predictions (predictionID, teamGameID, minerID, predictionDate, teamA, teamB, teamAodds, teamBodds, tieOdds, predictedOutcome, wager, outcome, canOverwrite) 
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            prediction["predictionID"],
                            prediction["teamGameID"],
                            prediction["minerID"],
                            prediction["predictionDate"],
                            prediction["teamA"],
                            prediction["teamB"],
                            prediction["teamAodds"],
                            prediction["teamBodds"],
                            prediction["tieOdds"],
                            prediction["predictedOutcome"],
                            prediction["wager"],
                            prediction["outcome"],
                            prediction["canOverwrite"],
                        ),
                    )
                except sqlite3.IntegrityError as e:
                    logging.warning(f"Failed to insert prediction {prediction['predictionID']}: {e}")
            cursor.connection.commit()

    def get_predictions(self):
        """
        Retrieve all predictions from the database.

        Returns:
            Dict[str, Dict]: A dictionary of predictions, keyed by prediction ID.

        Behavior:
            - Queries the database for all predictions
            - Constructs a dictionary of prediction data
        """
        predictions = {}
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM predictions")
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                predictions[row[0]] = {columns[i]: row[i] for i in range(len(columns))}
        return predictions

    def get_game_data(self):
        """
        Retrieve all inactive game data from the database.

        Returns:
            Dict[str, Dict]: A dictionary of game data, keyed by game ID.

        Behavior:
            - Queries the database for all inactive games
            - Constructs a dictionary of game data
        """
        game_data = {}
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM games WHERE active = 0")
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                game_data[row[0]] = {columns[i]: row[i] for i in range(len(columns))}
        return game_data

    def get_miner_stats(self, uid=None):
        """
        Retrieve stats for a specific miner.

        Args:
            uid (str, optional): The UID of the miner to retrieve stats for.

        Returns:
            Dict or None: A dictionary of miner stats if found, None otherwise.

        Behavior:
            - Queries the database for miner stats based on the provided UID
            - Constructs a dictionary of miner stats if found
        """
        logging.info(f"Getting miner stats for uid: {uid}")
        if uid is not None:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM miner_stats WHERE miner_uid = ?", (str(uid),))
                columns = [column[0] for column in cursor.description]
                row = cursor.fetchone()
                if row is None:
                    logging.warning(f"No miner stats found for uid: {uid}")
                    return None
                miner_stats = {columns[i]: row[i] for i in range(len(columns))}
                logging.info(f"Miner stats: {miner_stats}")
                return miner_stats
        else:
            logging.error("No UID specified")
            return None

    def insert_miner_stats(self, stats):
        """
        Insert miner stats into the database.

        Args:
            stats (Dict): A dictionary of miner stats to insert.

        Behavior:
            - Constructs an SQL INSERT statement from the provided stats
            - Executes the INSERT statement
        """
        columns = ", ".join(stats.keys())
        placeholders = ", ".join("?" * len(stats))
        values = tuple(stats.values())
        with self.db_manager.get_cursor() as cursor:
            cursor.execute(f"INSERT INTO miner_stats ({columns}) VALUES ({placeholders})", values)
            cursor.connection.commit()

    def update_miner_stats(self, wager, prediction_date, miner_uid):
        """
        Update miner stats in the database.

        Args:
            wager (float): The wager amount for the prediction.
            prediction_date (str): The date of the prediction.
            miner_uid (str): The UID of the miner to update.

        Behavior:
            - Updates miner cash, last prediction date, lifetime wager, and lifetime predictions
            - Commits the changes to the database
            - Reloads the miner stats after update
        """
        logging.info(f"Updating miner stats for miner_uid: {miner_uid}")
        logging.debug(f"Miner stats: {self.miner_stats}")
        new_miner_cash = self.miner_stats["miner_cash"] - wager
        new_miner_last_prediction_date = prediction_date
        new_miner_lifetime_wager = self.miner_stats["miner_lifetime_wager"] + wager
        new_miner_lifetime_predictions = self.miner_stats["miner_lifetime_predictions"] + 1

        with self.db_manager.get_cursor() as cursor:
            cursor.execute(
                """UPDATE miner_stats SET miner_cash = ?,
                                miner_last_prediction_date = ?, 
                                miner_lifetime_wager = ?, 
                                miner_lifetime_predictions = ? 
                                WHERE miner_uid = ?""",
                (
                    new_miner_cash,
                    new_miner_last_prediction_date,
                    new_miner_lifetime_wager,
                    new_miner_lifetime_predictions,
                    miner_uid,
                ),
            )
            cursor.connection.commit()
        self.miner_stats = self.get_miner_stats(miner_uid)
        logging.info(f"Updated miner stats: {self.miner_stats}")


class InteractiveTable:
    """
    Base class for interactive tables
    """

    def __init__(self, app):
        """
        Initialize the InteractiveTable.

        Args:
            app: The main Application instance.

        Behavior:
            - Sets up the text area for displaying options
            - Initializes the frame and box for layout
            - Sets up the initial selected index and options list
        """
        self.app = app
        self.text_area = TextArea(
            focusable=True,
            read_only=True,
            width=prompt_toolkit.layout.Dimension(preferred=70),
            height=prompt_toolkit.layout.Dimension(
                preferred=20
            ),
        )
        self.frame = Frame(self.text_area, style="class:frame")
        self.box = HSplit([self.frame])
        self.selected_index = 0
        self.options = []

    def update_text_area(self):
        """
        Update the text area with the current options and selection.

        Behavior:
            - Formats the options list with the current selection highlighted
            - Updates the text area content
        """
        lines = [
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options)
        ]
        self.text_area.text = "\n".join(lines)

    def handle_enter(self):
        """
        Handle the enter key press.

        Behavior:
            - To be implemented by subclasses
        """
        pass

    def move_up(self):
        """
        Move the selection up.

        Behavior:
            - Decrements the selected index if not at the top
            - Updates the text area
        """
        if self.selected_index > 0:
            self.selected_index -= 1
        self.update_text_area()

    def move_down(self):
        """
        Move the selection down.

        Behavior:
            - Increments the selected index if not at the bottom
            - Updates the text area
        """
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
        self.update_text_area()


class MainMenu(InteractiveTable):
    """
    Main menu for the miner CLI - 1st level menu
    """

    def __init__(self, app):
        """
        Initialize the MainMenu.

        Args:
            app: The main Application instance.

        Behavior:
            - Sets up the header and options for the main menu
            - Calls the parent class initializer
            - Updates the text area with initial content
        """
        super().__init__(app)

        self.header = Label(
            " BetTensor Miner Main Menu", style="bold"
        )
        self.options = [
            "View Submitted Predictions",
            "View Games and Make Predictions",
            "Select Miner",
            "Quit",
        ]
        self.update_text_area()

    def update_text_area(self):
        """
        Update the text area with the current miner stats and menu options.

        Behavior:
            - Formats the miner stats and menu options
            - Updates the text area content
        """
        header_text = self.header.text
        divider = "-" * len(header_text)

        # Miner stats
        miner_stats_text = (
            f" Miner Hotkey: {self.app.miner_stats['miner_hotkey']}\n"
            f" Miner UID: {self.app.miner_stats['miner_uid']}\n"
            f" Miner Rank: {self.app.miner_stats['miner_rank']}\n"
            f" Miner Cash: {self.app.miner_stats['miner_cash']}\n"
            f" Current Incentive: {self.app.miner_stats['miner_current_incentive']} τ per day\n"
            f" Last Prediction: {self.app.miner_stats['miner_last_prediction_date']}\n"
            f" Lifetime Earnings: ${self.app.miner_stats['miner_lifetime_earnings']}\n"
            f" Lifetime Wager Amount: {self.app.miner_stats['miner_lifetime_wager']}\n"
            f" Lifetime Wins: {self.app.miner_stats['miner_lifetime_wins']}\n"
            f" Lifetime Losses: {self.app.miner_stats['miner_lifetime_losses']}\n"
            f" Win/Loss Ratio: {self.app.miner_stats['miner_win_loss_ratio']:.2f}\n"
            f" Miner Status: {self.app.miner_stats['miner_status']}\n"
        )

        options_text = "\n".join(
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options)
        )

        self.text_area.text = (
            f"{header_text}\n{divider}\n{miner_stats_text}\n{divider}\n{options_text}"
        )

    def handle_enter(self):
        """
        Handle the enter key press in the main menu.

        Behavior:
            - Performs the action corresponding to the selected option
            - Changes view or exits the application based on the selection
        """
        if self.selected_index == 0:
            self.app.change_view(PredictionsList(self.app))
        elif self.selected_index == 1:
            self.app.change_view(GamesList(self.app))
        elif self.selected_index == 2:
            self.app.miner_uid = self.app.select_miner()
            self.app.load_miner_data()
            self.app.reload_data()
            self.update_text_area()
        elif self.selected_index == 3:
            graceful_shutdown(self.app, submit=True)

    def move_up(self):
        """
        Move the selection up in the main menu.

        Behavior:
            - Decrements the selected index if not at the top
            - Updates the text area
        """
        super().move_up()
        self.update_text_area()

    def move_down(self):
        """
        Move the selection down in the main menu.

        Behavior:
            - Increments the selected index if not at the bottom
            - Updates the text area
        """
        super().move_down()
        self.update_text_area()


class PredictionsList(InteractiveTable):
    def __init__(self, app):
        """
        Initialize the PredictionsList.

        Args:
            app: The main Application instance.

        Behavior:
            - Sets up the predictions list view
            - Loads and sorts predictions
            - Initializes pagination
        """
        super().__init__(app)
        app.reload_data()
        self.message = ""
        self.predictions_per_page = 25
        self.current_page = 0
        self.update_sorted_predictions()
        self.update_total_pages()
        self.update_options()
        self.update_text_area()

    def update_sorted_predictions(self):
        """
        Update the sorted predictions list.

        Behavior:
            - Sorts the predictions by prediction date (most recent first)
        """
        self.sorted_predictions = sorted(
            self.app.predictions.values(),
            key=lambda x: datetime.datetime.fromisoformat(x["predictionDate"]),
            reverse=True
        )

    def update_total_pages(self):
        """
        Update the total number of pages for predictions pagination.

        Behavior:
            - Calculates the total number of pages based on the number of predictions and predictions per page
            - Ensures the current page is within the valid range
        """
        self.total_pages = max(1, (len(self.sorted_predictions) + self.predictions_per_page - 1) // self.predictions_per_page)
        self.current_page = min(self.current_page, self.total_pages - 1)

    def update_options(self):
        """
        Update the options list for the predictions view.

        Behavior:
            - Calculates the start and end indices for the current page
            - Generates formatted prediction strings for the current page
            - Adds a "Go Back" option at the end
        """
        start_idx = self.current_page * self.predictions_per_page
        end_idx = min(start_idx + self.predictions_per_page, len(self.sorted_predictions))
        
        if not self.sorted_predictions:
            # Set default lengths if there are no predictions
            max_teamA_len = len("Team A")
            max_teamB_len = len("Team B")
            max_teamAodds_len = len("Team A Odds")
            max_teamBodds_len = len("Team B Odds")
            max_tieOdds_len = len("Tie Odds")
            max_prediction_len = len("Prediction")
            max_wager_amount_len = len("Wager Amount")
            max_outcome_len = len("Outcome")
            max_date_len = len("Prediction Date")
        else:
            # Calculate maximum widths for each column
            max_teamA_len = max(len("Team A"), max(len(pred["teamA"]) for pred in self.sorted_predictions))
            max_teamB_len = max(len("Team B"), max(len(pred["teamB"]) for pred in self.sorted_predictions))
            max_teamAodds_len = max(len("Team A Odds"), max(len(f"{pred['teamAodds']:.2f}") for pred in self.sorted_predictions))
            max_teamBodds_len = max(len("Team B Odds"), max(len(f"{pred['teamBodds']:.2f}") for pred in self.sorted_predictions))
            max_tieOdds_len = max(len("Tie Odds"), max(len(f"{pred['tieOdds']:.2f}") for pred in self.sorted_predictions))
            max_prediction_len = max(len("Prediction"), max(len(pred["predictedOutcome"]) for pred in self.sorted_predictions))
            max_wager_amount_len = max(len("Wager Amount"), max(len(f"{pred['wager']:.2f}") for pred in self.sorted_predictions))
            max_outcome_len = max(len("Outcome"), max(len(str(pred["outcome"])) for pred in self.sorted_predictions))
            max_date_len = max(len("Prediction Date"), max(len(self.format_prediction_date(pred["predictionDate"])) for pred in self.sorted_predictions))

        # Define the header with calculated widths
        self.header = Label(
            f"  {'Team A':<{max_teamA_len}} | {'Team B':<{max_teamB_len}} | {'Team A Odds':<{max_teamAodds_len}} | {'Team B Odds':<{max_teamBodds_len}} | {'Tie Odds':<{max_tieOdds_len}} | {'Prediction':<{max_prediction_len}} | {'Wager Amount':<{max_wager_amount_len}} | {'Outcome':<{max_outcome_len}} | {'Prediction Date':<{max_date_len}}",
            style="bold",
        )

        # Generate options for the current page
        self.options = [
            f"{pred['teamA']:<{max_teamA_len}} | {pred['teamB']:<{max_teamB_len}} | {pred['teamAodds']:<{max_teamAodds_len}.2f} | {pred['teamBodds']:<{max_teamBodds_len}.2f} | {pred['tieOdds']:<{max_tieOdds_len}.2f} | {pred['predictedOutcome']:<{max_prediction_len}} | {pred['wager']:<{max_wager_amount_len}.2f} | {str(pred['outcome']):<{max_outcome_len}} | {self.format_prediction_date(pred['predictionDate']):<{max_date_len}}"
            for pred in self.sorted_predictions[start_idx:end_idx]
        ]
        self.options.append("Go Back")

    def update_text_area(self):
        """
        Update the text area for the predictions view.

        Behavior:
            - Formats the header, options, and pagination information
            - Updates the text area content
        """
        header_text = self.header.text
        divider = "-" * len(header_text)
        if len(self.options) <= 1:  # Only "Go Back" is present
            options_text = "No predictions available."
        else:
            options_text = "\n".join(
                f"> {option}" if i == self.selected_index else f"  {option}"
                for i, option in enumerate(self.options[:-1])
            )
        go_back_text = f"\n\n{'>' if self.selected_index == len(self.options) - 1 else ' '} {self.options[-1]}"
        page_info = f"\nPage {self.current_page + 1}/{self.total_pages} (Use left/right arrow keys to navigate)"
        self.text_area.text = f"{header_text}\n{divider}\n{options_text}{go_back_text}{page_info}\n\n{self.message}"

    def format_prediction_date(self, date_string):
        """
        Format a prediction date string.

        Args:
            date_string (str): The date string to format.

        Returns:
            str: The formatted date string.

        Behavior:
            - Converts the date string to a datetime object
            - Formats the datetime object as "YYYY-MM-DD HH:MM"
        """
        dt = datetime.datetime.fromisoformat(date_string)
        return dt.strftime("%Y-%m-%d %H:%M")

    def handle_enter(self):
        """
        Handle the enter key press in the predictions view.

        Behavior:
            - If the "Go Back" option is selected, changes the view to the main menu
            - Otherwise, opens the prediction view for the selected prediction
        """
        if self.selected_index == len(self.options) - 1:  # Go Back
            self.app.change_view(MainMenu(self.app))
        else:
            selected_prediction = self.sorted_predictions[self.current_page * self.predictions_per_page + self.selected_index]

            # Find the corresponding game data
            game_id = selected_prediction.get("teamGameID")
            game_data = next(
                (game for game in self.app.games.values() if game["gameID"] == game_id),
                None,
            )

            # TODO - switch to prediction view (can't edit)

    def clear_message(self):
        """
        Clear the message in the predictions view.

        Behavior:
            - Sets the message to an empty string
            - Updates the text area
        """
        self.message = ""
        self.update_text_area()

    def move_up(self):
        """
        Move the selection up in the predictions view.

        Behavior:
            - Decrements the selected index if not at the top
            - Updates the text area
        """
        if self.selected_index > 0:
            self.selected_index -= 1
            self.update_text_area()

    def move_down(self):
        """
        Move the selection down in the predictions view.

        Behavior:
            - Increments the selected index if not at the bottom
            - Updates the text area
        """
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
            self.update_text_area()

    def move_left(self):
        """
        Move to the previous page in the predictions view.

        Behavior:
            - Decrements the current page if not on the first page
            - Resets the selected index to 0
            - Updates the options and text area
        """
        if self.current_page > 0:
            self.current_page -= 1
            self.selected_index = 0
            self.update_options()
            self.update_text_area()

    def move_right(self):
        """
        Move to the next page in the predictions view.

        Behavior:
            - Increments the current page if not on the last page
            - Resets the selected index to 0
            - Updates the options and text area
        """
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.selected_index = 0
            self.update_options()
            self.update_text_area()


class GamesList(InteractiveTable):
    def __init__(self, app):
        """
        Initialize the GamesList.

        Args:
            app: The main Application instance.

        Behavior:
            - Sets up the games list view
            - Loads and sorts games
            - Initializes pagination and filtering
        """
        super().__init__(app)
        app.reload_data()

        # Get unique sports from all games
        self.available_sports = sorted(set(game['sport'] for game in self.app.games.values()))
        self.current_filter = "All Sports"  # Default to no filter

        self.update_sorted_games()

        self.games_per_page = 25
        self.current_page = 0
        self.update_total_pages()

        self.update_options()
        self.update_text_area()

    def update_sorted_games(self):
        """
        Update the sorted games list.

        Behavior:
            - Sorts the games by event start date (earliest first)
            - Applies the current filter if not set to "All Sports"
        """
        if self.current_filter == "All Sports":
            self.sorted_games = sorted(
                self.app.games.values(),
                key=lambda x: datetime.datetime.fromisoformat(x["eventStartDate"])
            )
        else:
            self.sorted_games = sorted(
                [game for game in self.app.games.values() if game['sport'] == self.current_filter],
                key=lambda x: datetime.datetime.fromisoformat(x["eventStartDate"])
            )

    def update_total_pages(self):
        """
        Update the total number of pages for games pagination.

        Behavior:
            - Calculates the total number of pages based on the number of games and games per page
            - Ensures the current page is within the valid range
        """
        self.total_pages = max(1, (len(self.sorted_games) + self.games_per_page - 1) // self.games_per_page)
        self.current_page = min(self.current_page, self.total_pages - 1)

    def update_options(self):
        """
        Update the options list for the games view.

        Behavior:
            - Calculates the start and end indices for the current page
            - Generates formatted game strings for the current page
            - Adds a "Filter" and "Go Back" option at the end
        """
        start_idx = self.current_page * self.games_per_page
        end_idx = min(start_idx + self.games_per_page, len(self.sorted_games))
        
        if not self.sorted_games:
            # Set default lengths if there are no games
            max_sport_len = len("Sport")
            max_teamA_len = len("Team A")
            max_teamB_len = len("Team B")
            max_eventStartDate_len = len("Event Start Date (UTC)")
            max_teamAodds_len = len("Team A Odds")
            max_teamBodds_len = len("Team B Odds")
            max_tieOdds_len = len("Tie Odds")
        else:
            # Calculate maximum widths for each column based on data and header
            max_sport_len = max(
                max(len(game["sport"]) for game in self.sorted_games),
                len("Sport"),
            )
            max_teamA_len = max(
                max(len(game["teamA"]) for game in self.sorted_games),
                len("Team A"),
            )
            max_teamB_len = max(
                max(len(game["teamB"]) for game in self.sorted_games),
                len("Team B"),
            )
            max_eventStartDate_len = max(
                max(len(game["eventStartDate"]) for game in self.sorted_games),
                len("Event Start Date"),
            )
            max_teamAodds_len = max(
                max(
                    len(str(game.get("teamAodds", "")))
                    for game in self.sorted_games
                ),
                len("Team A Odds"),
            )
            max_teamBodds_len = max(
                max(
                    len(str(game.get("teamBodds", "")))
                    for game in self.sorted_games
                ),
                len("Team B Odds"),
            )
            max_tieOdds_len = max(
                max(
                    len(str(game.get("tieOdds", "")))
                    for game in self.sorted_games
                ),
                len("Tie Odds"),
            )

        # Define the header with calculated widths
        self.header = Label(
            f"  {'Sport':<{max_sport_len}} | {'Team A':<{max_teamA_len}} | {'Team B':<{max_teamB_len}} | {'Event Start Date (UTC)':<{max_eventStartDate_len}} | {'Team A Odds':<{max_teamAodds_len}} | {'Team B Odds':<{max_teamBodds_len}} | {'Tie Odds':<{max_tieOdds_len}} ",
            style="bold",
        )

        # Generate options for the current page
        self.options = [
            f"{game['sport']:<{max_sport_len}} | {game['teamA']:<{max_teamA_len}} | {game['teamB']:<{max_teamB_len}} | {self.format_event_start_date(game['eventStartDate']):<{max_eventStartDate_len}} | {str(game.get('teamAodds', '')):<{max_teamAodds_len}} | {str(game.get('teamBodds', '')):<{max_teamBodds_len}} | {str(game.get('tieOdds', '')):<{max_tieOdds_len}}"
            for game in self.sorted_games[start_idx:end_idx]
        ]
        # Add filter option and Go Back
        self.options.append(f"Filter: {self.current_filter}")
        self.options.append("Go Back")

    def update_text_area(self):
        """
        Update the text area for the games view.

        Behavior:
            - Formats the header, options, and pagination information
            - Updates the text area content
        """
        header_text = self.header.text
        divider = "-" * len(header_text)
        if len(self.options) == 2:  # Only "Filter" and "Go Back" are present
            options_text = "No games available."
        else:
            options_text = "\n".join(
                f"> {option}" if i == self.selected_index else f"  {option}"
                for i, option in enumerate(self.options[:-2])
            )
        current_time_text = f"\n\nCurrent Time (UTC): {datetime.datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M')}"
        page_info = f"\nPage {self.current_page + 1}/{self.total_pages} (Use left/right arrow keys to navigate)"
        go_back_text = (
            f"\n\n  {self.options[-1]}"
            if self.selected_index != len(self.options) - 1
            else f"\n\n> {self.options[-1]}"
        )
        filter_text = (
            f"\n\n  {self.options[-2]}"
            if self.selected_index != len(self.options) - 2
            else f"\n\n> {self.options[-2]}"
        )
        self.text_area.text = (
            f"{header_text}\n{divider}\n{options_text}{filter_text}{go_back_text}{current_time_text}{page_info}"
        )

    def handle_enter(self):
        """
        Handle the enter key press in the games view.

        Behavior:
            - If the "Go Back" option is selected, changes the view to the main menu
            - If the "Filter" option is selected, cycles through the available sports filters
            - Otherwise, opens the wager confirmation view for the selected game
        """
        if self.selected_index == len(self.options) - 1:  # Go Back
            self.app.change_view(MainMenu(self.app))
        elif self.selected_index == len(self.options) - 2:  # Filter option
            self.cycle_filter()
        else:
            # Get the currently selected game data from the sorted list
            game_index = self.current_page * self.games_per_page + self.selected_index
            if game_index < len(self.sorted_games):
                selected_game_data = self.sorted_games[game_index]
                # Change view to WagerConfirm, passing the selected game data
                self.app.change_view(WagerConfirm(self.app, selected_game_data, self))

    def cycle_filter(self):
        """
        Cycle through the available sports filters.

        Behavior:
            - Updates the current filter to the next available sport
            - If the end of the list is reached, cycles back to "All Sports"
            - Updates the sorted games, total pages, and text area
        """
        current_index = self.available_sports.index(self.current_filter) if self.current_filter != "All Sports" else -1
        next_index = (current_index + 1) % (len(self.available_sports) + 1)
        self.current_filter = self.available_sports[next_index] if next_index < len(self.available_sports) else "All Sports"
        self.update_sorted_games()
        self.update_total_pages()
        self.current_page = 0
        self.selected_index = 0
        self.update_options()
        self.update_text_area()

    def move_up(self):
        """
        Move the selection up in the games view.

        Behavior:
            - Decrements the selected index if not at the top
            - Updates the text area
        """
        if self.selected_index > 0:
            self.selected_index -= 1
            self.update_text_area()

    def move_down(self):
        """
        Move the selection down in the games view.

        Behavior:
            - Increments the selected index if not at the bottom
            - Updates the text area
        """
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
            self.update_text_area()

    def move_left(self):
        """
        Move to the previous page in the games view.

        Behavior:
            - Decrements the current page if not on the first page
            - Resets the selected index to 0
            - Updates the options and text area
        """
        if self.current_page > 0:
            self.current_page -= 1
            self.selected_index = 0
            self.update_options()
            self.update_text_area()

    def move_right(self):
        """
        Move to the next page in the games view.

        Behavior:
            - Increments the current page if not on the last page
            - Resets the selected index to 0
            - Updates the options and text area
        """
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.selected_index = 0
            self.update_options()
            self.update_text_area()

    def format_event_start_date(self, event_start_date):
        """
        Format an event start date string.

        Args:
            event_start_date (str): The event start date string to format.

        Returns:
            str: The formatted event start date string.

        Behavior:
            - Converts the event start date string to a datetime object
            - Converts the datetime object to UTC
            - Formats the datetime object as "YYYY-MM-DD HH:MM"
        """
        # Parse the ISO format date
        dt = datetime.datetime.fromisoformat(event_start_date)
        # Convert to UTC
        dt = dt.astimezone(pytz.utc)
        # Format it to the desired format
        return dt.strftime("%Y-%m-%d %H:%M")


class WagerConfirm(InteractiveTable):
    """
    Wager confirmation view
    """

    def __init__(self, app, game_data, previous_view, wager_amount=""):
        """
        Initialize the WagerConfirm view.

        Args:
            app: The main Application instance.
            game_data: Data for the game being wagered on.
            previous_view: The view to return to after confirmation.
            wager_amount: Initial wager amount (default is empty string).

        Behavior:
            - Sets up the wager confirmation view
            - Initializes the wager input field
            - Sets up options for confirming or canceling the wager
        """
        super().__init__(app)
        self.game_data = game_data
        self.previous_view = previous_view
        self.miner_cash = app.miner_cash
        self.selected_team = game_data["teamA"]  # Default to teamA
        self.wager_input = TextArea(
            text=str(wager_amount),  # Set the initial wager amount
            multiline=False,
            password=False,
            focusable=True,
        )
        self.options = [
            "Change Selected Team",
            "Enter Wager Amount",
            "Confirm Wager",
            "Go Back",
        ]
        self.confirmation_message = ""
        self.update_text_area()

    def update_text_area(self):
        """
        Update the text area for the wager confirmation view.

        Behavior:
            - Formats the game info, miner cash, selected team, wager amount, and options
            - Updates the text area content
        """
        game_info = (
            f" {self.game_data['sport']} | {self.game_data['teamA']} vs {self.game_data['teamB']} | {self.game_data['eventStartDate']} | "
            f"Team A Odds: {self.game_data['teamAodds']} | Team B Odds: {self.game_data['teamBodds']} | Tie Odds: {self.game_data['tieOdds']}"
        )
        cash_info = f"Miner's Cash: ${self.miner_cash}"
        selected_team_text = f"Selected Team: {self.selected_team}"
        wager_input_text = f"Wager Amount: {self.wager_input.text}"
        options_text = "\n".join(
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options)
        )
        self.text_area.text = f"{game_info}\n{cash_info}\n{selected_team_text}\n{wager_input_text}\n{options_text}\n\n{self.confirmation_message}"
        self.box = HSplit([self.text_area, self.wager_input])

    def handle_enter(self):
        """
        Handle the enter key press in the wager confirmation view.

        Behavior:
            - If the "Change Selected Team" option is selected, toggles the selected team
            - If the "Enter Wager Amount" option is selected, focuses the wager input field
            - If the "Confirm Wager" option is selected, validates and submits the wager
            - If the "Go Back" option is selected, returns to the previous view
        """
        if self.selected_index == 0:  # Change Selected Team
            self.toggle_selected_team()
        elif self.selected_index == 1:  # Enter Wager Amount
            self.focus_wager_input()
        elif self.selected_index == 2:  # Confirm Wager
            try:
                wager_amount = float(self.wager_input.text.strip())
                if wager_amount <= 0 or wager_amount > self.miner_cash:
                    raise ValueError("Invalid wager amount")
                # Add the prediction to unsubmitted_predictions
                prediction_id = str(uuid.uuid4())  # Generate a new UUID for each prediction
                self.app.unsubmitted_predictions[prediction_id] = {
                    "predictionID": prediction_id,
                    "teamGameID": self.game_data["externalID"],
                    "minerID": self.app.miner_uid,  # Use the currently selected miner's UID
                    "sport": self.game_data["sport"],
                    "teamA": self.game_data["teamA"],
                    "teamB": self.game_data["teamB"],
                    "eventStartDate": self.game_data["eventStartDate"],
                    "predictionDate": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "teamAodds": self.game_data["teamAodds"],
                    "teamBodds": self.game_data["teamBodds"],
                    "tieOdds": self.game_data["tieOdds"],
                    "predictedOutcome": self.selected_team,
                    "wager": wager_amount,
                    "canOverwrite": True,
                    "outcome": "",
                }
                self.app.miner_cash -= wager_amount
                self.confirmation_message = "Wager confirmed! Submitting Prediction to Validators..."
                self.update_text_area()
                self.app.submit_predictions()
                threading.Timer(2.0, lambda: self.app.change_view(self.previous_view)).start()
            except ValueError:
                self.wager_input.text = "Invalid amount. Try again."
                self.update_text_area()
        elif self.selected_index == 3:  # Go Back
            self.app.change_view(self.previous_view)

    def move_up(self):
        """
        Move the selection up in the wager confirmation view.

        Behavior:
            - Decrements the selected index if not at the top
            - Updates the text area
        """
        if self.selected_index > 0:
            self.selected_index -= 1
            self.update_text_area()

    def move_down(self):
        """
        Move the selection down in the wager confirmation view.

        Behavior:
            - Increments the selected index if not at the bottom
            - Updates the text area
        """
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
            self.update_text_area()

    def focus_wager_input(self):
        """
        Focus the wager input field.

        Behavior:
            - Sets the focus to the wager input field
        """
        self.app.layout.focus(self.wager_input)

    def blur_wager_input(self):
        """
        Blur the wager input field.

        Behavior:
            - Removes the focus from the wager input field
        """
        self.app.layout.focus(self.text_area)

    def handle_wager_input_enter(self):
        """
        Handle the enter key press in the wager input field.

        Behavior:
            - Blurs the wager input field
            - Moves the focus to the "Confirm Wager" option
            - Updates the text area
            - Ensures the focus is back on the text area
        """
        self.blur_wager_input()
        self.selected_index = 2  # Move focus to "Confirm Wager"
        self.update_text_area()
        self.app.layout.focus(self.text_area)  # Ensure focus is back on the text area

    def toggle_selected_team(self):
        """
        Toggle the selected team.

        Behavior:
            - Cycles through the available teams (teamA, teamB, and Tie if applicable)
            - Updates the text area
        """
        if self.selected_team == self.game_data["teamA"]:
            self.selected_team = self.game_data["teamB"]
        elif self.selected_team == self.game_data["teamB"] and self.game_data.get(
            "canTie", False
        ):
            self.selected_team = "Tie"
        else:
            self.selected_team = self.game_data["teamA"]
        self.update_text_area()


bindings = KeyBindings()


@bindings.add("up")
def _(event):
    """
    Handle the up arrow key press.

    Args:
        event: The key press event.

    Behavior:
        - Moves the selection up in the current view
        - Blurs the wager input field if in the WagerConfirm view
    """
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, "current_view"):
            custom_app.current_view.move_up()
            if isinstance(custom_app.current_view, WagerConfirm):
                custom_app.current_view.blur_wager_input()
    except AttributeError as e:
        logging.error(f"Failed to move up: {e}")
        print("Failed to move up:", str(e))


@bindings.add("down")
def _(event):
    """
    Handle the down arrow key press.

    Args:
        event: The key press event.

    Behavior:
        - Moves the selection down in the current view
        - Blurs the wager input field if in the WagerConfirm view
    """
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, "current_view"):
            custom_app.current_view.move_down()
            if isinstance(custom_app.current_view, WagerConfirm):
                custom_app.current_view.blur_wager_input()
    except AttributeError as e:
        logging.error(f"Failed to move down: {e}")
        print("Failed to move down:", str(e))


@bindings.add("enter")
def _(event):
    """
    Handle the enter key press.

    Args:
        event: The key press event.

    Behavior:
        - Handles the enter key press in the current view
        - If in the WagerConfirm view and the wager input field is focused, handles the wager input enter event
    """
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, "current_view"):
            if isinstance(custom_app.current_view, WagerConfirm):
                if custom_app.current_view.app.layout.has_focus(
                    custom_app.current_view.wager_input
                ):
                    custom_app.current_view.handle_wager_input_enter()
                else:
                    custom_app.current_view.handle_enter()
            else:
                custom_app.current_view.handle_enter()
    except AttributeError as e:
        logging.error(f"Failed to handle enter: {e}")
        print("Failed to handle enter:", str(e))


@bindings.add("q")
def _(event):
    """
    Handle the 'q' key press.

    Args:
        event: The key press event.

    Behavior:
        - Performs a graceful shutdown of the application
    """
    custom_app = event.app.custom_app
    graceful_shutdown(custom_app, submit=False)


@bindings.add("c-z", eager=True)
def _(event):
    """
    Handle the Ctrl+Z key press.

    Args:
        event: The key press event.

    Behavior:
        - Performs a graceful shutdown of the application
    """
    custom_app = event.app.custom_app
    graceful_shutdown(custom_app, submit=False)


@bindings.add("c-c", eager=True)
def _(event):
    """
    Handle the Ctrl+C key press.

    Args:
        event: The key press event.

    Behavior:
        - Performs a graceful shutdown of the application
    """
    custom_app = event.app.custom_app
    graceful_shutdown(custom_app, submit=False)


@bindings.add("left")
def _(event):
    """
    Handle the left arrow key press.

    Args:
        event: The key press event.

    Behavior:
        - Moves to the previous page in the GamesList view
    """
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, "current_view") and isinstance(custom_app.current_view, GamesList):
            custom_app.current_view.move_left()
    except AttributeError as e:
        logging.error(f"Failed to move left: {e}")
        print("Failed to move left:", str(e))


@bindings.add("right")
def _(event):
    """
    Handle the right arrow key press.

    Args:
        event: The key press event.

    Behavior:
        - Moves to the next page in the GamesList view
    """
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, "current_view") and isinstance(custom_app.current_view, GamesList):
            custom_app.current_view.move_right()
    except AttributeError as e:
        logging.error(f"Failed to move right: {e}")
        print("Failed to move right:", str(e))


def graceful_shutdown(app, submit: bool):
    """
    Perform a graceful shutdown of the application.

    Args:
        app: The main Application instance.
        submit (bool): Whether to submit predictions before shutting down.

    Behavior:
        - Submits predictions if specified
        - Exits the application
    """
    if submit:
        print("Submitting predictions...")
        logging.info("Submitting predictions...")
        app.submit_predictions()  # Call the method directly on the app instance
    app.app.exit()


def signal_handler(signal, frame):
    """
    Handle system signals for graceful shutdown.

    Args:
        signal: The received signal.
        frame: The current stack frame.

    Behavior:
        - Logs the received signal
        - Calls the graceful_shutdown function
    """
    logging.info(f"Signal received, shutting down gracefully...")
    graceful_shutdown(app)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    app = Application()
    app.run()
