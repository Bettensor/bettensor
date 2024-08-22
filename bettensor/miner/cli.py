import argparse
import json
import logging
import signal
import sqlite3
import time
import traceback
import uuid
import pytz
import subprocess
import bittensor as bt
import sys
import os
from datetime import datetime, timezone
from rich.console import Console, Group
from rich.table import Table, Column
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.style import Style
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from prompt_toolkit import Application as PromptApplication
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout as PromptLayout
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.filters import Condition
from prompt_toolkit.keys import Keys
import time
import random
from rich.live import Live
from rich.columns import Columns
from rich.text import Text
import psycopg2
from bettensor.miner.database.database_manager import DatabaseManager
from bettensor.miner.database.predictions import PredictionsHandler
from bettensor.miner.database.games import GamesHandler
from bettensor.miner.stats.miner_stats import MinerStateManager, MinerStatsHandler
from prompt_toolkit.application.current import get_app
from rich.console import Group
from rich.align import Align
import asyncio

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"cli_{timestamp}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Starting Bettensor CLI application")

# Define custom colors
DARK_GREEN = "dark_green"
LIGHT_GREEN = "green"
GOLD = "gold1"
LIGHT_GOLD = "yellow"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--miner_uid', type=str, help='Miner UID')
    args = parser.parse_args()

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    )

    with progress:
        main_task = progress.add_task("Starting up...", total=1)
        
        # Show the spinner immediately
        console.print("Initializing CLI...")
        
        # Create the application in a separate function to allow the spinner to update
        def create_app():
            return Application(progress, args.miner_uid)
        
        app = progress.update(main_task, advance=1, description="Creating application...", refresh=True)
        app = create_app()

    app.run()




class Application:
    def __init__(self, progress, miner_uid):
        self.console = Console()
        self.miner_uid = miner_uid
        self.cli_channel = f'cli:{uuid.uuid4()}'
        self.running = True
        self.current_view = "main_menu"
        self.page = 1
        self.items_per_page = 10
        self.kb = KeyBindings()
        self.setup_keybindings()
        self.cursor_position = 0
        self.search_query = ""
        self.search_mode = False
        self.wager_input_mode = False
        self.wager_input = ""
        self.predictions_search_query = ""
        self.predictions_search_mode = False
        self.last_key_press = None
        self.esc_pressed_once = False
        self.submission_message = None
        self.last_prediction_time = 0
        self.last_prediction_id = None
        self.confirmation_mode = False

        self.initialize(progress)

    def initialize(self, progress):
        with progress:
            task = progress.add_task("Initializing application...", total=6)

            progress.update(task, advance=1, description="Connecting to database...")
            self.init_database()

            progress.update(task, advance=1, description="Fetching available miners...")
            self.init_miners()

            progress.update(task, advance=1, description="Initializing miner state manager...")
            self.init_state_manager()

            progress.update(task, advance=1, description="Initializing predictions handler...")
            self.init_predictions_handler()

            progress.update(task, advance=1, description="Initializing games handler...")
            self.init_games_handler()

            progress.update(task, advance=1, description="Reloading miner data...")
            self.reload_data()

    def init_database(self):
        db_host = os.getenv('DB_HOST', 'localhost')
        db_name = os.getenv('DB_NAME', 'bettensor')
        db_user = os.getenv('DB_USER', 'root')
        db_password = os.getenv('DB_PASSWORD', 'bettensor_password')
        
        max_retries = 5
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.db_manager = DatabaseManager(db_name, db_user, db_password, db_host)
                # Test the connection
                self.db_manager.execute_query("SELECT 1")
                bt.logging.debug("Successfully connected to the database.")
                break
            except (psycopg2.OperationalError, psycopg2.Error) as e:
                if attempt < max_retries - 1:
                    bt.logging.warning(f"Failed to connect to the database (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    bt.logging.error("Failed to connect to the database after multiple attempts.")
                    raise ValueError("Error: Unable to connect to the database. Please check your database configuration and ensure the database is running.")

    def init_miners(self):
        self.available_miners = self.get_available_miners()

        if not self.available_miners:
            #self.console_print("[bold yellow]Warning: No miners found in the database. Some features may not work correctly.[/bold yellow]")
            self.available_miners = [{'miner_uid': 'default', 'miner_hotkey': 'default', 'miner_cash': 1000, 'miner_rank': 'N/A'}]

        self.miner_stats = {str(row['miner_uid']): row for row in self.available_miners}

        parser = argparse.ArgumentParser(description="BetTensor Miner CLI")
        parser.add_argument("--uid", help="Specify the miner UID to start with")
        args = parser.parse_args()
        self.current_miner_uid = args.uid if args.uid else self.get_saved_miner_uid()

        valid_miner_found = False
        for miner in self.available_miners:
            if str(miner['miner_uid']) == str(self.current_miner_uid):
                self.miner_hotkey = str(miner['miner_hotkey'])
                self.miner_uid = str(miner['miner_uid'])
                self.miner_cash = float(miner['miner_cash'])  # Explicitly store miner cash
                valid_miner_found = True
                break

        if not valid_miner_found:
            self.miner_hotkey = str(self.available_miners[0]['miner_hotkey'])
            self.miner_uid = str(self.available_miners[0]['miner_uid'])
            self.miner_cash = float(self.available_miners[0]['miner_cash'])  # Explicitly store miner cash

        self.save_miner_uid(self.miner_uid)

    def get_available_miners(self):
        self.console_print("Retrieving miners from database...")
        query = "SELECT miner_uid, miner_hotkey, miner_cash, miner_rank FROM miner_stats"
        try:
            result = self.db_manager.execute_query(query)
            logging.debug(f"Retrieved miners: {result}")
            return result
        except Exception as e:
            self.console_print(f"[bold red]Failed to retrieve miners: {e}[/bold red]")
            logging.error(f"Error retrieving miners: {e}")
            return []

    def init_state_manager(self):
        self.state_manager = MinerStateManager(self.db_manager, self.miner_hotkey, self.miner_uid)
        self.stats_handler = MinerStatsHandler(self.state_manager)
        self.state_manager.load_state()

    def init_predictions_handler(self):
        self.predictions_handler = PredictionsHandler(self.db_manager, self.state_manager, self.miner_hotkey)

    def init_games_handler(self):
        self.games_handler = GamesHandler(self.db_manager, self.predictions_handler)
        self.unsubmitted_predictions = {}

    def reload_data(self):
        """
        Reload all data for the current miner.
        """
        try:
            self.state_manager.load_state()
            self.miner_stats = self.state_manager.get_stats()
            self.miner_cash = float(self.miner_stats.get('miner_cash', 0))
            self.miner_lifetime_earnings = float(self.miner_stats.get('miner_lifetime_earnings', 0))
            self.miner_lifetime_wager = float(self.miner_stats.get('miner_lifetime_wager', 0))
            self.miner_lifetime_wins = int(self.miner_stats.get('miner_lifetime_wins', 0))
            self.miner_lifetime_losses = int(self.miner_stats.get('miner_lifetime_losses', 0))
            self.miner_win_loss_ratio = float(self.miner_stats.get('miner_win_loss_ratio', 0))
            self.miner_last_prediction_date = self.miner_stats.get('miner_last_prediction_date')
        except Exception as e:
            bt.logging.error(f"Error reloading data: {str(e)}")

    @staticmethod
    def format_date(date_str):
        if date_str is None:
            return "N/A"
        try:
            date = datetime.fromisoformat(date_str)
            return date.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return str(date_str)

    @staticmethod
    def format_event_start_date(event_start_date):
        if isinstance(event_start_date, str):
            dt = datetime.fromisoformat(event_start_date.replace('Z', '+00:00'))
        elif isinstance(event_start_date, datetime):
            dt = event_start_date
        else:
            return str(event_start_date)
        return dt.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def format_last_prediction_date(date_str):
        if date_str is None:
            return "N/A"
        try:
            date = datetime.fromisoformat(date_str)
            return date.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return str(date_str)

    def generate_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["header"].update(Panel("BetTensor Miner CLI", style=f"bold {GOLD}"))
        
        if self.current_view == "main_menu":
            layout["body"].update(self.generate_main_menu())
            footer_text = "m: Main Menu | p: Predictions | g: Games | n: Next Miner | q: Quit"
        elif self.current_view == "predictions":
            layout["body"].update(self.generate_predictions_view())
            if self.predictions_search_mode:
                footer_text = "Enter: Finish Search | Esc: Cancel Search | Type to filter predictions"
            else:
                footer_text = "m: Main Menu | ↑/↓: Navigate | ←/→: Change Page | s: Search | q: Quit"
        elif self.current_view == "games":
            layout["body"].update(self.generate_games_view())
            if self.search_mode:
                footer_text = "Enter: Finish Search | Esc: Cancel Search | Type to filter games"
            else:
                footer_text = "m: Main Menu | ↑/↓: Navigate | ←/→: Change Page | Enter: Select Game | s: Search | q: Quit"
        elif self.current_view == "enter_prediction":
            layout["body"].update(self.generate_enter_prediction_view())
            footer_text = "a/b/t: Select Outcome | w: Enter/Exit Wager | c: Confirm | Esc: Cancel | q: Quit"
        
        layout["footer"].update(Panel(footer_text, style=f"italic {LIGHT_GREEN}"))
        
        return layout

    def generate_main_menu(self):
        console_width = self.console.width
        console_height = self.console.height

        # Calculate available space for the table
        available_height = console_height - 10  # Adjust for header, footer, and padding
        available_width = console_width - 4  # Adjust for panel borders

        # Determine the number of rows and columns based on available space
        num_rows = min(11, available_height)  # We have 11 items to display
        num_columns = 2

        # Calculate column widths
        option_width = min(30, available_width // 3)
        value_width = available_width - option_width - 3  # 3 for padding and separator

        # Adjust font size based on available space
        if available_width < 80:
            option_style = f"dim {LIGHT_GREEN}"
            value_style = f"dim {GOLD}"
        elif available_width > 120:
            option_style = f"bold {LIGHT_GREEN}"
            value_style = f"bold {GOLD}"
        else:
            option_style = LIGHT_GREEN
            value_style = GOLD

        table = Table(show_header=False, box=box.ROUNDED, border_style=DARK_GREEN, expand=True)
        table.add_column("Option", style=option_style, width=option_width)
        table.add_column("Value", style=value_style, width=value_width)
        
        rows = [
            ("Miner Hotkey", self.miner_hotkey),
            ("Miner UID", self.miner_uid),
            ("Miner Cash", f"${self.miner_cash:.2f}"),
            ("Current Incentive", f"{self.stats_handler.get_current_incentive():.2f} τ per day"),
            ("Last Prediction", self.format_last_prediction_date(self.miner_last_prediction_date)),
            ("Lifetime Earnings", f"${self.miner_lifetime_earnings:.2f}"),
            ("Lifetime Wager Amount", f"${self.miner_lifetime_wager:.2f}"),
            ("Lifetime Wins", str(self.miner_lifetime_wins)),
            ("Lifetime Losses", str(self.miner_lifetime_losses)),
            ("Win/Loss Ratio", f"{self.miner_win_loss_ratio:.2f}")
        ]

        for row in rows[:num_rows]:
            table.add_row(row[0], row[1])

        return Panel(table, title="Miner Statistics", border_style=DARK_GREEN, expand=True)

    def generate_predictions_view(self):
        predictions = self.predictions_handler.get_predictions_with_teams(self.miner_uid)
        logging.debug(f"Retrieved {len(predictions)} predictions")
        
        if self.predictions_search_query:
            filtered_predictions = {k: v for k, v in predictions.items() if 
                           self.predictions_search_query.lower() in v['home'].lower() or 
                           self.predictions_search_query.lower() in v['away'].lower() or 
                           self.predictions_search_query.lower() in v['predictedoutcome'].lower() or 
                           self.predictions_search_query.lower() in v['outcome'].lower()}
            logging.debug(f"Filtered to {len(filtered_predictions)} predictions")
            predictions = filtered_predictions
        
        if not predictions:
            logging.debug("No predictions found matching the search criteria.")
            return Panel(
                Group(
                    Text(f"Miner UID: {self.miner_uid}", style=f"bold {GOLD}"),
                    Text("No predictions found matching the search criteria.", justify="center", style=LIGHT_GREEN)
                ),
                title="Predictions",
                border_style=DARK_GREEN
            )

        # Calculate max widths for each column
        max_widths = {
            "Date": max(len(self.format_date(pred['predictiondate'])) for pred in predictions.values()),
            "Game ID": max(len(pred['teamgameid']) for pred in predictions.values()),
            "Home": max(len(pred['home']) for pred in predictions.values()),
            "Away": max(len(pred['away']) for pred in predictions.values()),
            "Predicted Outcome": max(len(pred['predictedoutcome']) for pred in predictions.values()),
            "Wager": max(len(f"${pred['wager']:.2f}") for pred in predictions.values()),
            "Wager Odds": max(len(f"{pred['teamaodds']:.2f}") for pred in predictions.values()),  # Use teamaodds as a placeholder
            "Result": max(len(pred['outcome']) for pred in predictions.values()),
            "Payout": 10  # Assuming a reasonable width for payout
        }

        table = Table(box=box.ROUNDED, expand=True, border_style=DARK_GREEN)
        table.add_column("Date", style=LIGHT_GREEN, width=max_widths["Date"])
        table.add_column("Game ID", style=LIGHT_GREEN, width=max_widths["Game ID"])
        table.add_column("Home", style=GOLD, width=max_widths["Home"])
        table.add_column("Away", style=GOLD, width=max_widths["Away"])
        table.add_column("Predicted Outcome", style=LIGHT_GREEN, width=max_widths["Predicted Outcome"])
        table.add_column("Wager", style=LIGHT_GOLD, width=max_widths["Wager"])
        table.add_column("Wager Odds", style=LIGHT_GOLD, width=max_widths["Wager Odds"])
        table.add_column("Result", style=LIGHT_GREEN, width=max_widths["Result"])
        table.add_column("Payout", style=LIGHT_GOLD, width=max_widths["Payout"])

        start = (self.page - 1) * self.items_per_page
        end = start + self.items_per_page
        predictions_to_display = list(predictions.values())[start:end]
        logging.debug(f"Displaying predictions {start} to {end} (total: {len(predictions_to_display)})")
        for pred in predictions_to_display:
            # Calculate wager odds based on the predicted outcome
            if pred['predictedoutcome'] == pred['home']:
                wager_odds = pred['teamaodds']
            elif pred['predictedoutcome'] == pred['away']:
                wager_odds = pred['teambodds']
            else:
                wager_odds = pred['tieodds'] if 'tieodds' in pred else 'N/A'

            # Calculate payout
            if pred['outcome'] == 'Wager Won':
                payout = pred['wager'] * wager_odds
            elif pred['outcome'] == 'Wager Lost':
                payout = 0
            else:
                payout = 'Pending'

            table.add_row(
                self.format_date(pred['predictiondate']),
                pred['teamgameid'],
                pred['home'],
                pred['away'],
                pred['predictedoutcome'],
                f"${pred['wager']:.2f}",
                f"{wager_odds:.2f}" if isinstance(wager_odds, (int, float)) else str(wager_odds),
                pred['outcome'],
                f"${payout:.2f}" if isinstance(payout, (int, float)) else str(payout)
            )

        if self.predictions_search_mode:
            footer = f"Search: {self.predictions_search_query} | Page {self.page} of {max(1, (len(predictions) - 1) // self.items_per_page + 1)}"
        else:
            footer = f"Page {self.page} of {max(1, (len(predictions) - 1) // self.items_per_page + 1)}"
        return Panel(
            Group(
                Text(f"Miner UID: {self.miner_uid}", style=f"bold {GOLD}"),
                table,
                Text(footer, justify="center", style=LIGHT_GREEN)
            ),
            title="Predictions",
            border_style=DARK_GREEN
        )

    def generate_games_view(self):
        games = self.games_handler.get_active_games()
        if self.search_query:
            games = {k: v for k, v in games.items() if self.search_query.lower() in v.sport.lower() or 
                    self.search_query.lower() in v.teamA.lower() or 
                    self.search_query.lower() in v.teamB.lower()}
        
        if not games:
            return Panel(
                Group(
                    self.generate_miner_info(),
                    Text("No games found matching the search criteria.", justify="center", style=LIGHT_GREEN)
                ),
                title="Active Games",
                border_style=DARK_GREEN
            )

        max_widths = {
            "Sport": max((len(game.sport) for game in games.values()), default=10),
            "Team A": max((len(game.teamA) for game in games.values()), default=10),
            "Team B": max((len(game.teamB) for game in games.values()), default=10),
            "Start Date": len("YYYY-MM-DD HH:MM"),
            "Team A Odds": len("100.00"),
            "Team B Odds": len("100.00"),
            "Tie Odds": len("100.00")
        }

        table = Table(box=box.ROUNDED, expand=True, border_style=DARK_GREEN)
        table.add_column("Sport", style=LIGHT_GREEN, width=max_widths["Sport"])
        table.add_column("Team A", style=GOLD, width=max_widths["Team A"])
        table.add_column("Team B", style=GOLD, width=max_widths["Team B"])
        table.add_column("Start Date", style=LIGHT_GREEN, width=max_widths["Start Date"])
        table.add_column("Team A Odds", style=LIGHT_GOLD, width=max_widths["Team A Odds"])
        table.add_column("Team B Odds", style=LIGHT_GOLD, width=max_widths["Team B Odds"])
        table.add_column("Tie Odds", style=LIGHT_GOLD, width=max_widths["Tie Odds"])

        start = (self.page - 1) * self.items_per_page
        end = start + self.items_per_page
        for i, (game_id, game) in enumerate(list(games.items())[start:end]):
            style = "reverse" if i == self.cursor_position else ""
            table.add_row(
                game.sport,
                game.teamA,
                game.teamB,
                self.format_event_start_date(game.eventStartDate),
                f"{game.teamAodds:.2f}",
                f"{game.teamBodds:.2f}",
                f"{game.tieOdds:.2f}" if game.tieOdds is not None else "N/A",
                style=style
            )

        if self.search_mode:
            footer = f"Search: {self.search_query} | Page {self.page} of {max(1, (len(games) - 1) // self.items_per_page + 1)}"
        else:
            footer = f"Page {self.page} of {max(1, (len(games) - 1) // self.items_per_page + 1)}"

        return Panel(
            Group(
                self.generate_miner_info(),
                table,
                Text(footer, justify="center", style=LIGHT_GREEN)
            ),
            title="Active Games",
            border_style=DARK_GREEN
        )

    def enter_prediction_for_selected_game(self):
        games = list(self.games_handler.get_active_games().items())
        start = (self.page - 1) * self.items_per_page
        selected_game_id, selected_game = games[start + self.cursor_position]
        
        self.current_view = "enter_prediction"
        self.selected_game = selected_game
        self.prediction_outcome = None
        self.prediction_wager = None
        self.last_prediction_time = 0  # Reset the last prediction time when entering a new prediction
        self.last_prediction_id = None  # Reset the last prediction ID
        self.can_tie = selected_game.canTie  # Store the canTie value

    def generate_enter_prediction_view(self):
        if hasattr(self, 'confirmation_mode') and self.confirmation_mode:
            # Only display the confirmation message
            content = [Text(self.submission_message, style="bold green", justify="center")]
            return Panel(Group(*content), title="Prediction Confirmation", border_style=DARK_GREEN)
        
        table = Table(show_header=False, box=box.ROUNDED, border_style=DARK_GREEN)
        table.add_column("Field", style=LIGHT_GREEN)
        table.add_column("Value", style=GOLD)
        
        table.add_row("Sport", self.selected_game.sport)
        table.add_row("Team A", self.selected_game.teamA)
        table.add_row("Team B", self.selected_game.teamB)
        table.add_row("Start Date", self.format_event_start_date(self.selected_game.eventStartDate))
        table.add_row("Team A Odds", f"{self.selected_game.teamAodds:.2f}")
        table.add_row("Team B Odds", f"{self.selected_game.teamBodds:.2f}")
        table.add_row("Tie Odds", f"{self.selected_game.tieOdds:.2f}" if self.selected_game.tieOdds is not None else "N/A")
        table.add_row("Predicted Outcome", self.prediction_outcome or "Not selected")
        
        if self.wager_input_mode:
            table.add_row("Wager Amount", f"${self.wager_input}")
        else:
            table.add_row("Wager Amount", f"${self.prediction_wager:.2f}" if self.prediction_wager is not None else "Not entered")

        footer_text = "Press 'a' for Team A, 'b' for Team B, 't' for Tie, 'w' to enter/exit wager input, 'c' to confirm, 'esc' to cancel"
        if self.wager_input_mode:
            footer_text = f"Enter wager amount (press 'w' to confirm, backspace to delete). Current input: ${self.wager_input}"
        
        content = [
            self.generate_miner_info(),
            table
        ]
        
        if hasattr(self, 'submission_message') and self.submission_message:
            message_style = "bold green" if "submitted" in self.submission_message else "bold red"
            content.append(Text(self.submission_message, style=message_style))
        
        content.append(Text(footer_text, style=LIGHT_GREEN))
        
        return Panel(Group(*content), title="Enter Prediction", border_style=DARK_GREEN)

    def filter_games(self):
        sport = Prompt.ask("Enter sport to filter (or leave blank for all)")
        date = Prompt.ask("Enter date to filter (YYYY-MM-DD) (or leave blank for all)")
        
        games = self.games_handler.get_active_games()
        filtered_games = {}
        for game_id, game in games.items():
            if (not sport or game.sport.lower() == sport.lower()) and \
               (not date or game.eventStartDate.startswith(date)):
                filtered_games[game_id] = game
        
        self.games_handler.active_games = filtered_games
        self.current_view = "games"

    def console_print(self, message, style="bold"):
        self.console.print(message, style=f"{style} {LIGHT_GREEN}")

    def money_rain_animation(self):
        def get_money_row():
            return "".join(random.choices(["$", " ", " ", " "], k=self.console.width))

        rows = [get_money_row() for _ in range(self.console.height)]
        
        with Live(refresh_per_second=20, screen=True) as live:
            for _ in range(30):  # 30 frames at 20 FPS is 1.5 seconds
                live.update(Text("\n".join(rows), style=GOLD))
                rows = rows[1:] + [get_money_row()]
                time.sleep(0.05)

    def is_not_searching(self):
        return not (self.search_mode or self.predictions_search_mode)

    def setup_keybindings(self):
        @self.kb.add('q', filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.quit()
            event.app.exit()

        @self.kb.add('m', filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.current_view = "main_menu"
            self.page = 1
            self.cursor_position = 0

        @self.kb.add('p', filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.current_view = "predictions"
            self.page = 1
            self.cursor_position = 0

        @self.kb.add('g', filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.current_view = "games"
            self.page = 1
            self.cursor_position = 0

        @self.kb.add('n', filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.select_next_miner()

        @self.kb.add('f', filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.filter_games()

        @self.kb.add('s', filter=Condition(lambda: self.current_view == "games" and not self.search_mode))
        def _(event):
            self.search_mode = True
            self.search_query = ""

        @self.kb.add('s', filter=Condition(lambda: self.current_view == "predictions" and not self.predictions_search_mode))
        def _(event):
            self.predictions_search_mode = True
            self.predictions_search_query = ""

        @self.kb.add(Keys.Escape, filter=Condition(lambda: self.search_mode or self.predictions_search_mode))
        def _(event):
            self.search_mode = False
            self.predictions_search_mode = False
            self.search_query = ""
            self.predictions_search_query = ""
            self.page = 1
            self.cursor_position = 0

        @self.kb.add(Keys.Any, filter=Condition(lambda: self.search_mode or self.predictions_search_mode))
        def _(event):
            logging.debug(f"Key pressed in search mode: {repr(event.data)}")
            
            if event.data == Keys.ControlJ:  # Enter key
                logging.debug("Enter key pressed, exiting search mode")
                self.search_mode = False
                self.predictions_search_mode = False
            elif event.data in (Keys.Backspace, Keys.Delete, '\x7f'):  # Include '\x7f' for backspace
                logging.debug(f"Backspace or Delete key pressed: {repr(event.data)}")
                if self.search_mode and self.search_query:
                    self.search_query = self.search_query[:-1]
                    logging.debug(f"Updated search query: {self.search_query}")
                elif self.predictions_search_mode and self.predictions_search_query:
                    self.predictions_search_query = self.predictions_search_query[:-1]
                    logging.debug(f"Updated predictions search query: {self.predictions_search_query}")
            elif event.data and len(event.data) == 1 and event.data.isprintable():
                if self.search_mode:
                    self.search_query += event.data
                    logging.debug(f"Added character to search query: {self.search_query}")
                else:
                    self.predictions_search_query += event.data
                    logging.debug(f"Added character to predictions search query: {self.predictions_search_query}")
            else:
                logging.debug(f"Unrecognized key: {repr(event.data)}")
            
            self.page = 1
            self.cursor_position = 0
            event.app.invalidate()

        @self.kb.add(Keys.Backspace, filter=Condition(lambda: self.search_mode or self.predictions_search_mode))
        def _(event):
            logging.debug("Explicit Backspace key pressed")
            if self.search_mode and self.search_query:
                self.search_query = self.search_query[:-1]
                logging.debug(f"Updated search query: {self.search_query}")
            elif self.predictions_search_mode and self.predictions_search_query:
                self.predictions_search_query = self.predictions_search_query[:-1]
                logging.debug(f"Updated predictions search query: {self.predictions_search_query}")
            self.page = 1
            self.cursor_position = 0
            event.app.invalidate()

        @self.kb.add('up', filter=Condition(lambda: self.current_view in ["games", "predictions"]))
        def _(event):
            self.cursor_position = max(0, self.cursor_position - 1)

        @self.kb.add('down', filter=Condition(lambda: self.current_view in ["games", "predictions"]))
        def _(event):
            self.cursor_position = min(self.items_per_page - 1, self.cursor_position + 1)

        @self.kb.add('left', filter=Condition(lambda: self.current_view in ["games", "predictions"]))
        def _(event):
            self.page = max(1, self.page - 1)
            self.cursor_position = 0

        @self.kb.add('right', filter=Condition(lambda: self.current_view in ["games", "predictions"]))
        def _(event):
            self.page += 1
            self.cursor_position = 0

        @self.kb.add('enter', filter=Condition(lambda: self.current_view == "games"))
        def _(event):
            self.enter_prediction_for_selected_game()

        @self.kb.add('a', filter=Condition(lambda: self.current_view == "enter_prediction"))
        def _(event):
            self.prediction_outcome = self.selected_game.teamA

        @self.kb.add('b', filter=Condition(lambda: self.current_view == "enter_prediction"))
        def _(event):
            self.prediction_outcome = self.selected_game.teamB

        @self.kb.add('t', filter=Condition(lambda: self.current_view == "enter_prediction" and self.can_tie))
        def _(event):
            self.prediction_outcome = "Tie"

        @self.kb.add('w', filter=Condition(lambda: self.current_view == "enter_prediction"))
        def _(event):
            if self.wager_input_mode:
                # Exit wager input mode
                self.wager_input_mode = False
                try:
                    self.prediction_wager = float(self.wager_input)
                    #self.console_print(f"[bold green]Wager of ${self.prediction_wager:.2f} entered successfully[/bold green]")
                except ValueError:
                    #self.console_print("[bold red]Invalid wager amount. Please enter a number.[/bold red]")
                    self.prediction_wager = None
                finally:
                    self.wager_input = ""
            else:
                # Enter wager input mode
                self.wager_input_mode = True
                self.wager_input = ""
            event.app.invalidate()

        @self.kb.add(Keys.Any, filter=Condition(lambda: self.wager_input_mode))
        def _(event):
            logging.debug(f"Key pressed in wager input mode: {event.data}")
            if event.data in (Keys.Backspace, Keys.Delete, '\x7f'):
                logging.debug("Backspace or Delete key pressed")
                self.wager_input = self.wager_input[:-1]
                logging.debug(f"Updated wager input: {self.wager_input}")
            elif len(event.data) == 1 and (event.data.isdigit() or event.data == '.'):
                self.wager_input += event.data
                logging.debug(f"Added character to wager input: {self.wager_input}")
            event.app.invalidate()

        @self.kb.add(Keys.Escape, filter=Condition(lambda: self.current_view == "enter_prediction"))
        def _(event):
            if self.wager_input_mode:
                self.wager_input_mode = False
                self.wager_input = ""
                self.prediction_wager = None
            else:
                self.current_view = "games"
            self.esc_pressed_once = False
            event.app.invalidate()

        @self.kb.add('c', filter=Condition(lambda: self.current_view == "enter_prediction" and not self.wager_input_mode))
        def _(event):
            logging.info(f"'c' key pressed. Current view: {self.current_view}, Wager input mode: {self.wager_input_mode}")
            logging.info(f"Prediction outcome: {self.prediction_outcome}, Prediction wager: {self.prediction_wager}")
            if self.prediction_outcome and self.prediction_wager is not None:
                logging.info(f"Submitting prediction: {self.prediction_outcome}, {self.prediction_wager}")
                self.submit_prediction()
            else:
                logging.warning("Attempted to submit prediction without outcome or wager")
                #self.console_print("[bold red]Please select an outcome and enter a wager amount[/bold red]")
            event.app.invalidate()  # Force redraw

        @self.kb.add(Keys.Enter, filter=Condition(lambda: self.search_mode or self.predictions_search_mode))
        def _(event):
            logging.debug("Enter key pressed in search mode")
            if self.search_mode:
                self.search_mode = False
                self.enter_prediction_for_selected_game()
            elif self.predictions_search_mode:
                self.predictions_search_mode = False
            self.search_query = ""
            self.predictions_search_query = ""
            event.app.invalidate()

        @self.kb.add(Keys.Any, filter=Condition(lambda: hasattr(self, 'confirmation_mode') and self.confirmation_mode))
        def _(event):
            # Ignore all key presses during confirmation mode
            pass

    def select_next_miner(self):
        current_index = next((i for i, miner in enumerate(self.available_miners) if str(miner['miner_uid']) == str(self.miner_uid)), -1)
        next_index = (current_index + 1) % len(self.available_miners)
        next_miner = self.available_miners[next_index]
        self.miner_uid = str(next_miner['miner_uid'])
        self.miner_hotkey = str(next_miner['miner_hotkey'])
        self.save_miner_uid(self.miner_uid)
        self.reload_miner_data()
        #self.console_print(f"[bold green]Switched to miner with UID: {self.miner_uid}[/bold green]")

    def reload_miner_data(self):
        try:
            # Fetch the latest miner stats from the database
            miner_stats = self.db_manager.execute_query(
                "SELECT * FROM miner_stats WHERE miner_uid = %s",
                (self.miner_uid,)
            )
            if miner_stats:
                miner_stats = miner_stats[0]  # Assuming the query returns a single row
                self.miner_cash = float(miner_stats.get('miner_cash', 0))
                self.miner_lifetime_earnings = float(miner_stats.get('miner_lifetime_earnings', 0))
                self.miner_lifetime_wager = float(miner_stats.get('miner_lifetime_wager', 0))
                self.miner_lifetime_wins = int(miner_stats.get('miner_lifetime_wins', 0))
                self.miner_lifetime_losses = int(miner_stats.get('miner_lifetime_losses', 0))
                self.miner_win_loss_ratio = float(miner_stats.get('miner_win_loss_ratio', 0))
                self.miner_last_prediction_date = miner_stats.get('miner_last_prediction_date')
            else:
                bt.logging.warning(f"No stats found for miner with UID: {self.miner_uid}")
        except Exception as e:
            bt.logging.error(f"Error reloading miner data: {str(e)}")

    def quit(self):
        bt.logging.info("Received signal 2. Shutting down...")
        bt.logging.info("Stopping miner")
        self.state_manager.save_state()
        bt.logging.info("Miner stopped")
        bt.logging.info("Exiting due to signal")
        sys.exit(0)

    def get_saved_miner_uid(self):
        """
        Retrieve the saved miner UID from file.
        If the file doesn't exist, return None.
        """
        file_path = 'current_miner_uid.txt'
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            bt.logging.warning(f"{file_path} not found. Will use the first available miner.")
            return None

    def save_miner_uid(self, uid):
        """
        Save the current miner UID to a file.
        """
        file_path = 'current_miner_uid.txt'
        with open(file_path, 'w') as f:
            f.write(str(uid))
        #bt.logging.info(f"Saved miner UID {uid} to {file_path}")

    def run(self):
        try:
            layout = PromptLayout(Window(content=FormattedTextControl(self.get_formatted_text)))
            app = PromptApplication(layout=layout, key_bindings=self.kb, full_screen=True)
            app.run()
        except Exception as e:
            self.console_print(f"[bold red]An error occurred: {e}[/bold red]")
            self.console_print(f"[bold yellow]Please check the log file: {log_file}[/bold yellow]")
            logging.error(f"Unhandled exception: {e}", exc_info=True)

    def get_formatted_text(self):
        
        layout = self.generate_layout()
        console = Console(width=self.console.width, height=self.console.height)
        with console.capture() as capture:
            console.print(layout)
        logging.debug(f"Last key press: {self.last_key_press}")
        return ANSI(capture.get())

    def clear_console(self):
        # For Windows
        if os.name == 'nt':
            _ = os.system('cls')
        # For Unix/Linux/MacOS
        else:
            _ = os.system('clear')

    def submit_prediction(self):
        if self.prediction_outcome and self.prediction_wager is not None:
            current_time = time.time()
            if current_time - self.last_prediction_time < 1:  # 1 second cooldown
                self.submission_message = "Please wait before submitting another prediction."
                return

            prediction = {
                'predictionID': str(uuid.uuid4()),
                'teamGameID': self.selected_game.externalId,
                'predictionDate': datetime.now(timezone.utc).isoformat(),
                'predictedOutcome': self.prediction_outcome,
                'teamA': self.selected_game.teamA,
                'teamB': self.selected_game.teamB,
                'wager': self.prediction_wager,
                'teamAodds': self.selected_game.teamAodds,
                'teamBodds': self.selected_game.teamBodds,
                'tieOdds': self.selected_game.tieOdds,
                'outcome': 'unfinished'
            }

            result = self.predictions_handler.add_prediction(prediction)

            if result['status'] == 'success':
                self.submission_message = f"Prediction submitted: {self.prediction_outcome} with wager ${self.prediction_wager:.2f}"
                self.last_prediction_time = current_time
                self.last_prediction_id = prediction['predictionID']
                self.reload_miner_data()
                
                # Set a flag to indicate we're in confirmation mode
                self.confirmation_mode = True
                
                # Schedule return to games view after 2 seconds
                app = get_app()
                app.invalidate()
                app.create_background_task(self.delayed_return_to_games())
            else:
                self.submission_message = f"Failed to submit prediction: {result['message']}"
        else:
            self.submission_message = "Please select an outcome and enter a wager amount."
        
        # Force a redraw
        get_app().invalidate()

    async def delayed_return_to_games(self):
        await asyncio.sleep(2)  # Wait for 2 seconds
        self.current_view = "games"
        self.submission_message = None
        self.confirmation_mode = False
        get_app().invalidate()  # Force a redraw

    def generate_miner_info(self):
        current_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        miner_info = Table.grid(padding=(0, 1))
        miner_info.add_column(style="bold " + LIGHT_GREEN, justify="right")
        miner_info.add_column(style=GOLD, justify="left")
        miner_info.add_row("Miner UID:", self.miner_uid)
        miner_info.add_row("Cash:", f"${self.miner_cash:.2f}")
        miner_info.add_row("Last Prediction:", self.format_last_prediction_date(self.miner_last_prediction_date))
        miner_info.add_row("Current Time:", current_time_utc)

        # Wrap the table in a panel with a fixed width
        panel = Panel(
            miner_info,
            title="Miner Info",
            border_style=DARK_GREEN,
            width=60,  # Increased width to accommodate the time
            expand=False
        )

        # Center the panel
        centered_panel = Align.center(panel)

        return Group(centered_panel)

if __name__ == "__main__":
    main()