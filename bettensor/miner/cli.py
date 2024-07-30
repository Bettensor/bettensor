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
from prompt_toolkit.shortcuts import clear
from rich.console import Console
from rich.table import Table
from prompt_toolkit.application import Application as PTApplication
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import Frame, TextArea, Label
from prompt_toolkit.styles import Style
from prompt_toolkit.layout.containers import Window, HSplit
import logging
import os
from bettensor.miner.database.database_manager import get_db_manager
from bettensor.miner.database.predictions import PredictionsHandler
from bettensor.miner.database.games import GamesHandler
from bettensor.miner.stats.miner_stats import MinerStateManager
import threading
import os
import sys
import subprocess
import atexit
from prompt_toolkit.output import Output

# Create logs directory if it doesn't exist
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging for CLI
cli_log_file = os.path.join(log_dir, f"cli_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(cli_log_file),
        logging.StreamHandler()  # This will still print to console
    ]
)

# Create a logger for the CLI
cli_logger = logging.getLogger("cli")

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
            raise ValueError("Error: Database not found. Please start the miner first.")

        self.available_miners = self.get_available_miners()
        
        if not self.available_miners:
            raise ValueError("Error: No miners found in the database. Please start the miner first.")

        # Load the saved miner index
        self.current_miner_index = self.get_saved_miner_index()
        self.miner_uid = self.available_miners[self.current_miner_index][0]
        self.miner_hotkey = self.available_miners[self.current_miner_index][1]

        self.state_manager = MinerStateManager(self.db_manager, self.miner_hotkey, self.miner_uid)
        
        self.predictions_handler = PredictionsHandler(self.db_manager, self.state_manager, self.miner_uid)
        self.games_handler = GamesHandler(self.db_manager)

        # Initialize unsubmitted_predictions
        self.unsubmitted_predictions = {}

        self.reload_data()
        self.running = True
        self.bindings = self.setup_key_bindings()
        self.layout = self.setup_layout()
        self.app = PTApplication(
            layout=self.layout,
            key_bindings=self.bindings,
            full_screen=True,
            style=global_style,
        )
        self.app.custom_app = self
        atexit.register(self.cleanup)

    def setup_key_bindings(self):
        kb = KeyBindings()

        @kb.add('c-c')
        @kb.add('c-q')
        def _(event):
            self.quit()

        @kb.add('up')
        def _(event):
            if hasattr(self, 'current_view'):
                self.current_view.move_up()

        @kb.add('down')
        def _(event):
            if hasattr(self, 'current_view'):
                self.current_view.move_down()

        @kb.add('enter')
        def _(event):
            if hasattr(self, 'current_view'):
                self.current_view.handle_enter()

        @kb.add('left')
        def _(event):
            if hasattr(self, 'current_view') and isinstance(self.current_view, GamesList):
                self.current_view.move_left()

        @kb.add('right')
        def _(event):
            if hasattr(self, 'current_view') and isinstance(self.current_view, GamesList):
                self.current_view.move_right()

        return kb

    def setup_layout(self):
        self.current_view = MainMenu(self)
        return Layout(self.current_view.box)

    def cleanup(self):
        if self.app and self.app.output:
            self.app.output.reset_attributes()
            self.app.output.enable_autowrap()
            self.app.output.quit_alternate_screen()
            self.app.output.flush()
        os.system('reset')

    def quit(self):
        cli_logger.info("Initiating shutdown...")
        self.running = False
        try:
            self.state_manager.save_state()  # Save state before exiting
            cli_logger.info(f"Final miner cash: {self.state_manager.get_stats()['miner_cash']}")
        except Exception as e:
            cli_logger.error(f"Error during shutdown: {e}")
            cli_logger.error(traceback.format_exc())
        finally:
            try:
                if self.app.is_running:
                    self.app.exit()
            except Exception as e:
                cli_logger.error(f"Error during application exit: {e}")
                cli_logger.error(traceback.format_exc())

        # Ensure terminal is reset
        self.cleanup()

        # Restart the application if we're selecting a new miner
        if self.current_view and isinstance(self.current_view, MainMenu) and self.current_view.selected_index == 2:
            python = sys.executable
            os.execl(python, python, *sys.argv)
        else:
            # If not restarting, exit explicitly
            sys.exit(0)

    def run(self):
        def run_app():
            try:
                self.app.run()
            except Exception as e:
                cli_logger.error(f"Error in app: {e}")
                cli_logger.error(traceback.format_exc())
            finally:
                self.running = False

        app_thread = threading.Thread(target=run_app, daemon=True)
        app_thread.start()

        try:
            while self.running:
                if not app_thread.is_alive():
                    break
                app_thread.join(0.1)
        except KeyboardInterrupt:
            cli_logger.info("Keyboard interrupt received. Shutting down...")
        finally:
            self.quit()

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

    def select_next_miner(self):
        """
        Rotate to the next available miner and restart the application.

        Behavior:
            - Cycles through available miners
            - Saves the new miner index
            - Restarts the entire application
        """
        self.current_miner_index = (self.current_miner_index + 1) % len(self.available_miners)
        
        # Save the current miner index to a file
        with open('current_miner_index.txt', 'w') as f:
            f.write(str(self.current_miner_index))
        
        # Quit the application, which will trigger a restart
        self.quit()

    @staticmethod
    def get_saved_miner_index():
        """
        Retrieve the saved miner index from file.
        """
        try:
            with open('current_miner_index.txt', 'r') as f:
                return int(f.read().strip())
        except FileNotFoundError:
            return 0

    def reload_data(self):
        """
        Reload all data for the current miner.

        Behavior:
            - Retrieves predictions, game data, and miner stats from the database
            - Initializes default values if no stats are found
            - Sets up active games and unsubmitted predictions
        """
        self.check_unsubmitted_predictions()
        self.predictions = self.predictions_handler.get_predictions(self.miner_uid)  # Add miner_uid here
        self.games = self.games_handler.get_active_games()
        self.reload_miner_stats()
        
        if self.miner_stats is None:
            cli_logger.warning(f"No stats found for miner {self.miner_uid}. Initializing new miner row.")
            self.state_manager.load_state()
            self.miner_stats = self.state_manager.get_stats()

        self.active_games = {}
        self.check_unsubmitted_predictions()
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
        self.reload_miner_stats()  # Reload stats before changing view
        self.current_view = new_view
        self.layout.container = new_view.box
        self.app.invalidate()

        # If changing to MainMenu, update miner data
        if isinstance(new_view, MainMenu):
            new_view.update_text_area()

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
            raise ValueError(f"Database not initialized properly, restart your miner first: {e}")

    def check_unsubmitted_predictions(self):
        pass

    def submit_predictions(self):
        self.check_unsubmitted_predictions()
        for prediction_id, prediction in self.unsubmitted_predictions.items():
            try:
                self.predictions_handler.add_prediction(prediction)
                self.reload_miner_stats()  # Reload stats after each prediction
            except Exception as e:
                cli_logger.error(f"Failed to submit prediction {prediction_id}: {str(e)}")
        self.check_unsubmitted_predictions()
        self.unsubmitted_predictions.clear()
        self.check_unsubmitted_predictions()
        self.reload_miner_stats()  # Reload stats after all predictions are submitted

    def get_predictions(self):
        """
        Retrieve all predictions from the database for the current miner.

        Returns:
            Dict[str, Dict]: A dictionary of predictions, keyed by prediction ID.

        Behavior:
            - Queries the database for all predictions for the current miner
            - Constructs a dictionary of prediction data
        """
        return self.predictions_handler.get_predictions(self.miner_uid)

    def get_game_data(self):
        """
        Retrieve all inactive game data from the database.

        Returns:
            Dict[str, Dict]: A dictionary of game data, keyed by game ID.

        Behavior:
            - Queries the database for all inactive games
            - Constructs a dictionary of game data
        """
        return self.games_handler.get_active_games()

    def get_miner_stats(self):
        """
        Retrieve stats for the current miner.

        Returns:
            Dict: A dictionary of miner stats.

        Behavior:
            - Retrieves the current miner stats from the state manager
        """
        return self.state_manager.get_stats()

    def update_miner_stats(self, wager, prediction_date):
        self.state_manager.update_on_prediction({'wager': wager, 'predictionDate': prediction_date})
        self.reload_data()

    def reload_miner_stats(self):
        self.miner_stats = self.state_manager.get_stats()
        self.miner_cash = self.miner_stats["miner_cash"]


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
            - If the "Go Back" option is selected, changes the view to the main menu
            - Otherwise, does nothing (for now)
        """
        if self.selected_index == len(self.options) - 1:  # Go Back
            self.app.change_view(MainMenu(self.app))
        
        

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
        app.reload_miner_stats()  # Reload stats when initializing MainMenu

        self.header = Label(
            " BetTensor Miner Main Menu", style="bold"
        )
        self.options = [
            "View Submitted Predictions",
            "View Games and Make Predictions",
            "Select Next Miner (This will trigger app restart, please be patient)",
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
        self.app.reload_miner_stats()  # Reload stats before updating text area
        header_text = self.header.text
        divider = "-" * len(header_text)

        # Miner stats
        miner_stats_text = (
            f" Miner Hotkey: {self.app.miner_stats['miner_hotkey']}\n"
            f" Miner UID: {self.app.miner_stats['miner_uid']}\n"
            f" Miner Rank: {self.app.miner_stats.get('miner_rank', 'N/A')}\n"
            f" Miner Cash: {self.app.miner_stats['miner_cash']:.2f}\n"
            f" Current Incentive: {self.app.miner_stats['miner_current_incentive']:.2f} τ per day\n"
            f" Last Prediction: {self.app.miner_stats.get('miner_last_prediction_date', 'N/A')}\n"
            f" Lifetime Earnings: ${self.app.miner_stats['miner_lifetime_earnings']:.2f}\n"
            f" Lifetime Wager Amount: {self.app.miner_stats['miner_lifetime_wager']:.2f}\n"
            f" Lifetime Wins: {self.app.miner_stats['miner_lifetime_wins']}\n"
            f" Lifetime Losses: {self.app.miner_stats['miner_lifetime_losses']}\n"
            f" Win/Loss Ratio: {self.app.miner_stats['miner_win_loss_ratio']:.2f}\n"
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
            self.show_loading_message()
            self.app.select_next_miner()
        elif self.selected_index == 3:
            self.app.quit()

    def show_loading_message(self):
        self.text_area.text = "Loading next miner... Please wait."
        self.app.app.invalidate()

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
        app.reload_miner_stats()  # Reload stats when initializing PredictionsList
        app.reload_data()
        self.message = ""
        self.predictions_per_page = 25
        self.current_page = 0
        self.update_sorted_predictions()
        self.update_total_pages()
        self.update_options()
        self.selected_index = len(self.options) - 1  # Set cursor to "Go Back"
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
            - Calculates dynamic column widths based on data
            - Formats predictions data with proper alignment and separators
        """
        start_idx = self.current_page * self.predictions_per_page
        end_idx = min(start_idx + self.predictions_per_page, len(self.sorted_predictions))
        
        # Calculate maximum widths for each column
        max_date_len = max(len("Prediction Date"), max(len(self.format_date(pred["predictionDate"])) for pred in self.sorted_predictions))
        max_prediction_len = max(len("Predicted Outcome"), max(len(str(pred["predictedOutcome"])) for pred in self.sorted_predictions))
        max_teamA_len = max(len("Team A"), max(len(str(pred["teamA"])) for pred in self.sorted_predictions))
        max_teamB_len = max(len("Team B"), max(len(str(pred["teamB"])) for pred in self.sorted_predictions))
        max_wager_amount_len = max(len("Wager Amount"), max(len(self.format_odds(pred["wager"])) for pred in self.sorted_predictions))
        max_teamAodds_len = max(len("Team A Odds"), max(len(self.format_odds(pred["teamAodds"])) for pred in self.sorted_predictions))
        max_teamBodds_len = max(len("Team B Odds"), max(len(self.format_odds(pred["teamBodds"])) for pred in self.sorted_predictions))
        max_tieOdds_len = max(len("Tie Odds"), max(len(self.format_odds(pred["tieOdds"])) for pred in self.sorted_predictions))
        max_outcome_len = max(len("Outcome"), max(len(self.format_outcome(pred["outcome"])) for pred in self.sorted_predictions))

        # Define the header with calculated widths, adding a space at the beginning for cursor alignment
        self.header = (
            f"  {'Prediction Date':<{max_date_len}} | "
            f"{'Predicted Outcome':<{max_prediction_len}} | "
            f"{'Team A':<{max_teamA_len}} | "
            f"{'Team B':<{max_teamB_len}} | "
            f"{'Wager Amount':<{max_wager_amount_len}} | "
            f"{'Team A Odds':<{max_teamAodds_len}} | "
            f"{'Team B Odds':<{max_teamBodds_len}} | "
            f"{'Tie Odds':<{max_tieOdds_len}} | "
            f"{'Outcome':<{max_outcome_len}}"
        )

        # Generate options for the current page
        self.options = []
        for pred in self.sorted_predictions[start_idx:end_idx]:
            self.options.append(
                f"{self.format_date(pred['predictionDate']):<{max_date_len}} | "
                f"{pred['predictedOutcome']:<{max_prediction_len}} | "
                f"{pred['teamA']:<{max_teamA_len}} | "
                f"{pred['teamB']:<{max_teamB_len}} | "
                f"{self.format_odds(pred['wager']):<{max_wager_amount_len}} | "
                f"{self.format_odds(pred['teamAodds']):<{max_teamAodds_len}} | "
                f"{self.format_odds(pred['teamBodds']):<{max_teamBodds_len}} | "
                f"{self.format_odds(pred['tieOdds']):<{max_tieOdds_len}} | "
                f"{self.format_outcome(pred['outcome']):<{max_outcome_len}}"
            )
        self.options.append("Go Back")

    def update_text_area(self):
        """
        Update the text area for the predictions view.

        Behavior:
            - Formats the header, options, and pagination information
            - Updates the text area content
        """
        self.app.reload_miner_stats()  # Reload stats before updating text area
        header_text = self.header
        divider = "-" * len(header_text)
        if len(self.options) <= 1:  # Only "Go Back" is present
            options_text = "No predictions available."
        else:
            options_text = "\n".join(
                f"{'>' if i == self.selected_index else ' '} {option}"
                for i, option in enumerate(self.options)
            )
        page_info = f"\nPage {self.current_page + 1}/{self.total_pages} (Use left/right arrow keys to navigate)"
        self.text_area.text = f"{header_text}\n{divider}\n{options_text}{page_info}\n\n{self.message}"

    def handle_enter(self):
        """
        Handle the enter key press in the predictions view.

        Behavior:
            - Changes the view to the main menu (since cursor is always on "Go Back")
        """
        self.app.change_view(MainMenu(self.app))

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
        Overridden to prevent cursor movement.
        """
        pass

    def move_down(self):
        """
        Overridden to prevent cursor movement.
        """
        pass

    def move_left(self):
        """
        Move to the previous page in the predictions view.

        Behavior:
            - Decrements the current page if not on the first page
            - Updates the options and text area
        """
        if self.current_page > 0:
            self.current_page -= 1
            self.update_options()
            self.selected_index = len(self.options) - 1  # Keep cursor on "Go Back"
            self.update_text_area()

    def move_right(self):
        """
        Move to the next page in the predictions view.

        Behavior:
            - Increments the current page if not on the last page
            - Updates the options and text area
        """
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_options()
            self.selected_index = len(self.options) - 1  # Keep cursor on "Go Back"
            self.update_text_area()

    def format_odds(self, value):
        """
        Format odds or wager values, handling both string and float types.

        Args:
            value: The value to format (can be string or float).

        Returns:
            str: The formatted value as a string.
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return f"{value:.2f}"
        else:
            return str(value)

    def format_outcome(self, outcome):
        """
        Format the outcome string.
        """
        return outcome  # We're now returning the outcome as-is

    def format_date(self, date_string):
        date_obj = datetime.datetime.fromisoformat(date_string)
        return date_obj.strftime("%Y-%m-%d %H:%M:%S")

import datetime  # Modify this import at the top of the file

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
        app.reload_miner_stats()  # Reload stats when initializing GamesList
        app.reload_data()
        self.available_sports = sorted(set(game.sport for game in self.app.games.values()))
        self.current_filter = "All Sports"
        self.update_sorted_games()
        self.games_per_page = 25
        self.current_page = 0
        self.update_total_pages()
        self.update_options()
        self.update_text_area()

    def update_options(self):
        """
        Update the options list for the games view.

        Behavior:
            - Calculates dynamic column widths based on data
            - Formats games data with proper alignment and separators
        """
        start_idx = self.current_page * self.games_per_page
        end_idx = min(start_idx + self.games_per_page, len(self.sorted_games))
        
        # Calculate maximum widths for each column
        max_sport_len = max(len("Sport"), max(len(game.sport) for game in self.sorted_games))
        max_teamA_len = max(len("Team A"), max(len(game.teamA) for game in self.sorted_games))
        max_teamB_len = max(len("Team B"), max(len(game.teamB) for game in self.sorted_games))
        max_eventStartDate_len = max(len("Event Start Date"), max(len(self.format_event_start_date(game.eventStartDate)) for game in self.sorted_games))
        max_teamAodds_len = max(len("Team A Odds"), max(len(self.format_odds(game.teamAodds)) for game in self.sorted_games))
        max_teamBodds_len = max(len("Team B Odds"), max(len(self.format_odds(game.teamBodds)) for game in self.sorted_games))
        max_tieOdds_len = max(len("Tie Odds"), max(len(self.format_odds(game.tieOdds)) for game in self.sorted_games))

        # Define the header with calculated widths, adding a space at the beginning for cursor alignment
        self.header = (
            f"  {'Sport':<{max_sport_len}} | "
            f"{'Team A':<{max_teamA_len}} | "
            f"{'Team B':<{max_teamB_len}} | "
            f"{'Event Start Date':<{max_eventStartDate_len}} | "
            f"{'Team A Odds':<{max_teamAodds_len}} | "
            f"{'Team B Odds':<{max_teamBodds_len}} | "
            f"{'Tie Odds':<{max_tieOdds_len}}"
        )

        # Generate options for the current page
        self.options = []
        for game in self.sorted_games[start_idx:end_idx]:
            self.options.append(
                f"{game.sport:<{max_sport_len}} | "
                f"{game.teamA:<{max_teamA_len}} | "
                f"{game.teamB:<{max_teamB_len}} | "
                f"{self.format_event_start_date(game.eventStartDate):<{max_eventStartDate_len}} | "
                f"{self.format_odds(game.teamAodds):<{max_teamAodds_len}} | "
                f"{self.format_odds(game.teamBodds):<{max_teamBodds_len}} | "
                f"{self.format_odds(game.tieOdds):<{max_tieOdds_len}}"
            )
        self.options.append(f"Filter: {self.current_filter}")
        self.options.append("Go Back")

    def format_odds(self, value):
        """
        Format odds values, handling both string and float types.

        Args:
            value: The value to format (can be string or float).

        Returns:
            str: The formatted value as a string.
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return f"{value:.2f}"
        else:
            return str(value)

    def update_sorted_games(self):
        """
        Update the sorted games list.

        Behavior:
            - Sorts the games by event start date (earliest first)
            - Applies the current filter if not set to "All Sports"
            - Filters out games that have already occurred
        """
        current_time = datetime.datetime.now(datetime.timezone.utc)
        if self.current_filter == "All Sports":
            self.sorted_games = sorted(
                [game for game in self.app.games.values() if self.parse_date(game.eventStartDate) > current_time],
                key=lambda x: self.parse_date(x.eventStartDate)
            )
        else:
            self.sorted_games = sorted(
                [game for game in self.app.games.values() if game.sport == self.current_filter and self.parse_date(game.eventStartDate) > current_time],
                key=lambda x: self.parse_date(x.eventStartDate)
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

    def update_text_area(self):
        """
        Update the text area for the games view.

        Behavior:
            - Formats the header, options, and pagination information
            - Updates the text area content
        """
        self.app.reload_miner_stats()  # Reload stats before updating text area
        header_text = self.header
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
        dt = self.parse_date(event_start_date)
        return dt.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def parse_date(date_string):
        """
        Parse a date string into a datetime object.

        Args:
            date_string (str): The date string to parse.

        Returns:
            datetime.datetime: The parsed datetime object.

        Behavior:
            - Converts the date string to a datetime object
            - Ensures the datetime is in UTC
        """
        return datetime.datetime.fromisoformat(date_string.replace('Z', '+00:00'))




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
        app.reload_miner_stats()  # Reload stats when initializing WagerConfirm
        self.game_data = game_data
        self.previous_view = previous_view
        self.miner_cash = app.miner_stats["miner_cash"]
        self.selected_team = game_data.teamA  # Default to teamA
        self.wager_input = TextArea(
            text=str(wager_amount),
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
        self.app.reload_miner_stats()  # Reload stats before updating text area
        game_info = (
            f" {self.game_data.sport} | {self.game_data.teamA} vs {self.game_data.teamB} | {self.game_data.eventStartDate} | "
            f"Team A Odds: {self.game_data.teamAodds} | Team B Odds: {self.game_data.teamBodds} | Tie Odds: {self.game_data.tieOdds}"
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
        if self.selected_index == 0:  # Change Selected Team
            self.toggle_selected_team()
        elif self.selected_index == 1:  # Enter Wager Amount
            self.focus_wager_input()
        elif self.selected_index == 2:  # Confirm Wager
            try:
                wager_amount = float(self.wager_input.text.strip())
                if wager_amount <= 0 or wager_amount > self.miner_cash:
                    raise ValueError("Invalid wager amount")
                
                prediction_id = str(uuid.uuid4())
                prediction = {
                    "predictionID": prediction_id,
                    "teamGameID": self.game_data.externalId,
                    "predictionDate": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "predictedOutcome": self.selected_team,
                    "wager": wager_amount,
                    "teamAodds": self.game_data.teamAodds,
                    "teamBodds": self.game_data.teamBodds,
                    "tieOdds": self.game_data.tieOdds,
                    "outcome": "Unfinished",
                    "teamA": self.game_data.teamA,
                    "teamB": self.game_data.teamB
                }
                self.app.unsubmitted_predictions[prediction_id] = prediction
                bt.logging.info(f"Added prediction to unsubmitted_predictions: {prediction}")
                
                self.app.submit_predictions()
                self.app.change_view(MainMenu(self.app))
            except ValueError as e:
                self.confirmation_message = str(e)
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
        if self.selected_team == self.game_data.teamA:
            self.selected_team = self.game_data.teamB
        elif self.selected_team == self.game_data.teamB and self.game_data.canTie:
            self.selected_team = "Tie"
        else:
            self.selected_team = self.game_data.teamA
        self.update_text_area()

if __name__ == "__main__":
    app = None
    try:
        app = Application()
        app.run()
    except Exception as e:
        bt.logging.error(f"Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())  # Add this line to get the full traceback
