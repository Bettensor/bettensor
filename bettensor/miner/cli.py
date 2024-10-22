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
import sys
import os
import warnings
from eth_utils.exceptions import ValidationError

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Network .* does not have a valid ChainId.*")


warnings.filterwarnings(
    "ignore",
    message="Network 345 with name 'Yooldo Verse Mainnet' does not have a valid ChainId.*",
)
warnings.filterwarnings(
    "ignore",
    message="Network 12611 with name 'Astar zkEVM' does not have a valid ChainId.*",
)

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
import bittensor as bt
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
from huggingface_hub import hf_hub_download
import torch

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"cli_{timestamp}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Starting Bettensor CLI application")

DARK_GREEN = "dark_green"
LIGHT_GREEN = "green"
GOLD = "gold1"
LIGHT_GOLD = "yellow"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=str, help="Miner UID to start with")
    args = parser.parse_args()

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    )

    with progress:
        main_task = progress.add_task("Starting up...", total=1)

        console.print("Initializing CLI...")

        def create_app():
            return Application(progress, args.uid)

        app = progress.update(
            main_task, advance=1, description="Creating application...", refresh=True
        )
        app = create_app()

    app.run()


class Application:
    def __init__(self, progress, miner_uid=None):
        self.console = Console()
        self.uid = miner_uid
        self.cli_channel = f"cli:{uuid.uuid4()}"
        self.running = True
        self.current_view = "main_menu"
        self.page = 1
        self.cursor_position = 0
        self.items_per_page = 10
        self.db_manager = None
        self.miner_uid = None
        self.miner_hotkey = None
        self.state_manager = None
        self.stats_handler = None
        self.predictions_handler = None
        self.games_handler = None
        self.available_miners = []

        # Miner stats
        self.miner_current_tier = 1
        self.miner_rank = 0
        self.miner_current_entropy_score = 0
        self.miner_current_clv_score = 0
        self.miner_current_composite_score = 0
        self.miner_current_sortino_ratio = 0
        self.miner_current_clv_avg = 0
        self.miner_current_incentive = 0

        self.miner_cash = 0
        self.miner_lifetime_earnings = 0
        self.miner_lifetime_wager = 0
        self.miner_lifetime_predictions = 0
        self.miner_lifetime_wins = 0
        self.miner_lifetime_losses = 0
        self.miner_lifetime_roi = 0
        self.miner_win_loss_ratio = 0
        self.miner_last_prediction_date = None
        

        # local active status
        self.miner_is_active = False


        self.last_key_press = None
        self.search_mode = False
        self.predictions_search_mode = False
        self.wager_input_mode = False
        self.search_query = ""
        self.predictions_search_query = ""
        self.wager_input = ""
        self.prediction_outcome = None
        self.prediction_wager = None
        self.esc_pressed_once = False
        self.submission_message = None
        self.confirmation_mode = False
        self.layout = None
        self.application = None
        self.log_messages = []
        self.log_message_times = []
        self.log_display_time = 10  #change log display time here

        self.kb = KeyBindings()
        self.setup_keybindings()
        self.reset_layout()

        with progress:
            task = progress.add_task("Initializing application...", total=6)

            progress.update(task, advance=1, description="Connecting to database...")
            self.init_database()

            progress.update(task, advance=1, description="Fetching available miners...")
            self.init_miners()

            if self.miner_uid and self.miner_uid != "default":
                progress.update(
                    task, advance=1, description="Initializing miner state manager..."
                )
                self.init_state_manager()

                progress.update(
                    task, advance=1, description="Initializing predictions handler..."
                )
                self.init_predictions_handler()

                progress.update(
                    task, advance=1, description="Initializing games handler..."
                )
                self.init_games_handler()

                progress.update(task, advance=1, description="Reloading miner data...")
                self.reload_miner_data()
            else:
                progress.update(
                    task,
                    advance=4,
                    description="No valid miner available. Skipping miner-specific initializations...",
                )
                bt.logging.warning(
                    "No valid miner available. The application will run with limited functionality."
                )

        self.start_log_cleaner()

    def init_database(self):
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "bettensor")
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASSWORD", "bettensor_password")

        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                self.db_manager = DatabaseManager(
                    db_name, db_user, db_password, db_host
                )

                self.db_manager.execute_query("SELECT 1")
                bt.logging.debug("Successfully connected to the database.")
                break
            except (psycopg2.OperationalError, psycopg2.Error) as e:
                if attempt < max_retries - 1:
                    bt.logging.warning(
                        f"Failed to connect to the database (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    bt.logging.error(
                        "Failed to connect to the database after multiple attempts."
                    )
                    raise ValueError(
                        "Error: Unable to connect to the database. Please check your database configuration and ensure the database is running."
                    )

    def init_miners(self):
        self.available_miners = self.get_available_miners()
        if not self.available_miners:
            if not hasattr(self, 'miner_cash') or self.miner_cash is None:
                self.miner_cash = 1000
            self.miner_uid = "default"
            self.miner_hotkey = "default"
            self.miner_is_active = False
        else:
            if self.uid:
                self.current_miner_uid = self.uid
            else:
                self.current_miner_uid = self.get_saved_miner_uid()

            valid_miner_found = False
            for miner in self.available_miners:
                if str(miner["miner_uid"]) == str(self.current_miner_uid):
                    self.miner_hotkey = str(miner["miner_hotkey"])
                    self.miner_uid = str(miner["miner_uid"])
                    self.miner_cash = float(miner["miner_cash"])
                    self.miner_rank = int(miner["miner_rank"])
                    self.miner_current_entropy_score = float(miner["miner_current_entropy_score"])
                    self.miner_current_clv_avg = float(miner["miner_current_clv_avg"])
                    self.miner_current_composite_score = float(miner["miner_current_composite_score"])
                    self.miner_current_sortino_ratio = float(miner["miner_current_sortino_ratio"])
                    self.miner_current_incentive = float(miner["miner_current_incentive"])
                    self.miner_current_tier = int(miner["miner_current_tier"])
                    self.miner_lifetime_roi = float(miner["miner_lifetime_roi"])
                    valid_miner_found = True
                    break

            if not valid_miner_found:
                self.miner_hotkey = str(self.available_miners[0]["miner_hotkey"])
                self.miner_uid = str(self.available_miners[0]["miner_uid"])
                if not hasattr(self, 'miner_cash') or self.miner_cash is None:
                    self.miner_cash = float(self.available_miners[0]["miner_cash"])
                self.miner_is_active = self.available_miners[0].get("is_active", False)

        self.save_miner_uid(self.miner_uid)

    def get_available_miners(self):
        self.add_log_message("Retrieving miners from database...")
        query = """
        SELECT ms.miner_uid, ms.miner_hotkey, ms.miner_cash, ms.miner_rank, ms.miner_current_entropy_score, ms.miner_current_clv_avg, ms.miner_current_composite_score, ms.miner_current_sortino_ratio, ms.miner_current_incentive,
                ms.miner_current_tier,ms.miner_lifetime_roi,

               CASE WHEN ma.last_active_timestamp > NOW() - INTERVAL '5 minutes' THEN TRUE ELSE FALSE END as is_active
        FROM miner_stats ms
        LEFT JOIN miner_active ma ON ms.miner_uid::text = ma.miner_uid::text
        """
        try:
            result = self.db_manager.execute_query(query)
            logging.debug(f"Retrieved miners: {result}")
            return result
        except Exception as e:
            self.add_log_message(f"Failed to retrieve miners: {e}", "ERROR")
            logging.error(f"Error retrieving miners: {e}")
            return []

    def init_state_manager(self):
        if self.miner_uid and self.miner_uid != "default":
            self.state_manager = MinerStateManager(
                self.db_manager, self.miner_hotkey, self.miner_uid
            )
            self.stats_handler = MinerStatsHandler(self.state_manager)
            
        else:
            self.state_manager = None
            self.stats_handler = None
            bt.logging.warning(
                "No valid miner selected. State manager and stats handler not initialized."
            )

    def init_predictions_handler(self):
        if self.miner_uid is None or self.miner_uid == "default":
            bt.logging.warning(
                "Miner UID is not set. Skipping predictions handler initialization."
            )
            self.predictions_handler = None
        else:
            self.predictions_handler = PredictionsHandler(
                self.db_manager, self.state_manager, self.miner_hotkey
            )
            self.db_manager.initialize_default_model_params(self.miner_uid)

    def init_games_handler(self):
        if self.miner_uid and self.miner_uid != "default" and self.predictions_handler:
            self.games_handler = GamesHandler(self.db_manager, self.predictions_handler)
            self.unsubmitted_predictions = {}
        else:
            self.games_handler = None
            self.unsubmitted_predictions = {}

    def reload_miner_data(self):
        if not self.miner_uid or self.miner_uid == "default" or not self.db_manager:
            bt.logging.warning(
                "No valid miner or database manager. Skipping miner data reload."
            )
            return

        try:
            miner_stats = self.db_manager.execute_query(
                "SELECT * FROM miner_stats WHERE miner_uid = %s", (self.miner_uid,)
            )
            if miner_stats:
                miner_stats = miner_stats[0]
                self.miner_cash = float(miner_stats.get("miner_cash", 0))
                self.miner_lifetime_earnings = float(
                    miner_stats.get("miner_lifetime_earnings", 0)
                )
                self.miner_lifetime_wager = float(
                    miner_stats.get("miner_lifetime_wager_amount", 0)
                )
                self.miner_lifetime_wins = int(
                    miner_stats.get("miner_lifetime_wins", 0)
                )
                self.miner_lifetime_losses = int(
                    miner_stats.get("miner_lifetime_losses", 0)
                )
                self.miner_win_loss_ratio = float(
                    miner_stats.get("miner_win_loss_ratio", 0)
                )
                self.miner_last_prediction_date = miner_stats.get(
                    "miner_last_prediction_date"
                )
            else:
                bt.logging.warning(
                    f"No stats found for miner with UID: {self.miner_uid}"
                )
        except Exception as e:
            self.add_log_message(f"Error reloading miner data: {str(e)}", "ERROR")

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
            dt = datetime.fromisoformat(event_start_date.replace("Z", "+00:00"))
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
        try:
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body", ratio=1),
                Layout(name="logs", size=5),
                Layout(name="footer", size=3),
            )

            layout["header"].update(Panel("BetTensor Miner CLI", style=f"bold {GOLD}"))
            
            body_content = None
            if self.current_view == "main_menu":
                body_content = self.generate_main_menu()
                footer_text = "m: Main Menu | p: Predictions | g: Games | n: Next Miner | q: Quit"
            elif self.current_view == "predictions":
                body_content = self.generate_predictions_view()
                footer_text = "m: Main Menu | ↑/↓: Navigate | ←/→: Change Page | s: Search | q: Quit" if not self.predictions_search_mode else "Enter: Finish Search | Esc: Cancel Search | Type to filter predictions"
            elif self.current_view == "games":
                body_content = self.generate_games_view()
                footer_text = "m: Main Menu | ↑/↓: Navigate | ←/→: Change Page | Enter: Select Game | s: Search | q: Quit" if not self.search_mode else "Enter: Finish Search | Esc: Cancel Search | Type to filter games"
            elif self.current_view == "enter_prediction":
                body_content = self.generate_enter_prediction_view()
                footer_text = "a/b/t: Select Outcome | w: Enter/Exit Wager | c: Confirm | Esc: Cancel | q: Quit"
            else:
                body_content = Panel("Invalid view")
                footer_text = "q: Quit"

            if body_content is not None:
                layout["body"].update(body_content)
            else:
                layout["body"].update(Panel("Error: Unable to generate content"))

            log_content = "\n".join(self.log_messages[-3:])
            layout["logs"].update(Panel(log_content, title="Logs", border_style="blue"))

            layout["footer"].update(Panel(footer_text, style=f"italic {LIGHT_GREEN}"))
            return layout
        except Exception as e:
            logging.error(f"Error in generate_layout: {str(e)}")
            return Layout(Panel(f"Error generating layout: {str(e)}"))

    def generate_main_menu(self):
        console_width = self.console.width
        console_height = self.console.height

        available_height = console_height - 10
        available_width = console_width - 4

        num_rows = min(12, available_height)
        num_columns = 2

        option_width = min(30, available_width // 3)
        value_width = available_width - option_width - 3

        if available_width < 80:
            option_style = f"dim {LIGHT_GREEN}"
            value_style = f"dim {GOLD}"
        elif available_width > 120:
            option_style = f"bold {LIGHT_GREEN}"
            value_style = f"bold {GOLD}"
        else:
            option_style = LIGHT_GREEN
            value_style = GOLD

        table = Table(
            show_header=False, box=box.ROUNDED, border_style=DARK_GREEN, expand=True
        )
        table.add_column("Option", style=option_style, width=option_width)
        table.add_column("Value", style=value_style, width=value_width)

        is_active = (
            self.db_manager.is_miner_active(self.miner_uid) if self.miner_uid else False
        )
        active_status = "Active" if is_active else "Inactive"
        active_style = value_style if is_active else "bold red"

        rows = [
            ("Miner Hotkey", self.miner_hotkey or "N/A"),
            ("Miner UID", self.miner_uid or "N/A"),
            ("Miner Status", active_status),
            ("Miner Cash", f"${self.miner_cash:.2f}"),
            (
                "Current Incentive",
                f"{self.miner_current_incentive:.2f} τ per day"
                if self.miner_current_incentive
                else "N/A",
            ),
            (
                "Last Prediction",
                self.format_last_prediction_date(self.miner_last_prediction_date),
            ),
            ("Tier", str(self.miner_current_tier)),
            ("Entropy Score (Daily)", f"{self.miner_current_entropy_score:.2f}"),
            ("CLV Avg. (Daily)", f"{self.miner_current_clv_avg:.2f}"),
            ("Risk Score (Daily)", f"{self.miner_current_sortino_ratio:.2f}"),
            ("Composite Score (Tier Score)", f"{self.miner_current_composite_score:.2f}"),
            ("Lifetime ROI (%)", f"{self.miner_lifetime_roi:.2f}"),
            ("Lifetime Earnings", f"${self.miner_lifetime_earnings:.2f}"),
            ("Lifetime Wager Amount", f"${self.miner_lifetime_wager:.2f}"),
            ("Lifetime Wins", str(self.miner_lifetime_wins)),
            ("Lifetime Losses", str(self.miner_lifetime_losses)),
            ("Win/Loss Ratio", f"{self.miner_win_loss_ratio:.2f}"),
        ]

        for i, row in enumerate(rows[:num_rows]):
            if i == 2:
                table.add_row(row[0], Text(row[1], style=active_style))
            else:
                table.add_row(row[0], row[1])

        return Panel(
            table, title="Miner Statistics", border_style=DARK_GREEN, expand=True
        )

    def generate_predictions_view(self):
        if not self.predictions_handler or not self.miner_uid:
            return Panel(
                Text(
                    "No predictions available. Please select a valid miner.",
                    style=LIGHT_GREEN,
                ),
                title="Predictions",
                border_style=DARK_GREEN,
            )

        predictions = self.predictions_handler.get_predictions_with_teams(
            self.miner_uid
        )
        logging.debug(f"Retrieved {len(predictions)} predictions")

        if self.predictions_search_query:
            filtered_predictions = {
                k: v
                for k, v in predictions.items()
                if self.predictions_search_query.lower() in v["team_a"].lower()
                or self.predictions_search_query.lower() in v["team_b"].lower()
                or self.predictions_search_query.lower()
                in v["predicted_outcome"].lower()
                or self.predictions_search_query.lower() in v["outcome"].lower()
            }
            logging.debug(f"Filtered to {len(filtered_predictions)} predictions")
            predictions = filtered_predictions

        if not predictions:
            logging.debug("No predictions found matching the search criteria.")
            return Panel(
                Group(
                    Text(f"Miner UID: {self.miner_uid}", style=f"bold {GOLD}"),
                    Text(
                        "No predictions found matching the search criteria.",
                        justify="center",
                        style=LIGHT_GREEN,
                    ),
                ),
                title="Predictions",
                border_style=DARK_GREEN,
            )

        max_widths = {
            "Date": max(
                len(self.format_date(pred["prediction_date"]))
                for pred in predictions.values()
            ),
            "Game ID": max(len(pred["game_id"]) for pred in predictions.values()),
            "Home": max(len(pred["team_a"]) for pred in predictions.values()),
            "Away": max(len(pred["team_b"]) for pred in predictions.values()),
            "Predicted Outcome": max(
                len(pred["predicted_outcome"]) for pred in predictions.values()
            ),
            "Wager": max(len(f"${pred['wager']:.2f}") for pred in predictions.values()),
            "Wager Odds": max(
                len(f"{pred['team_a_odds']:.2f}") for pred in predictions.values()
            ),
            "Result": max(len(pred["outcome"]) for pred in predictions.values()),
            "Payout": 10,
            "Sent": max(
                len(str(pred["validators_sent_to"])) for pred in predictions.values()
            ),
            "Confirmed": max(
                len(str(pred["validators_confirmed"])) for pred in predictions.values()
            ),
        }

        table = Table(box=box.ROUNDED, expand=True, border_style=DARK_GREEN)
        table.add_column("Date", style=LIGHT_GREEN, width=max_widths["Date"])
        table.add_column("Game ID", style=LIGHT_GREEN, width=max_widths["Game ID"])
        table.add_column("Home", style=GOLD, width=max_widths["Home"])
        table.add_column("Away", style=GOLD, width=max_widths["Away"])
        table.add_column(
            "Predicted Outcome",
            style=LIGHT_GREEN,
            width=max_widths["Predicted Outcome"],
        )
        table.add_column("Wager", style=LIGHT_GOLD, width=max_widths["Wager"])
        table.add_column("Wager Odds", style=LIGHT_GOLD, width=max_widths["Wager Odds"])
        table.add_column("Result", style=LIGHT_GREEN, width=max_widths["Result"])
        table.add_column("Payout", style=LIGHT_GOLD, width=max_widths["Payout"])
        table.add_column("Sent", style=LIGHT_GREEN, width=max_widths["Sent"])
        table.add_column("Confirmed", style=LIGHT_GREEN, width=max_widths["Confirmed"])

        start = (self.page - 1) * self.items_per_page
        end = start + self.items_per_page
        predictions_to_display = list(predictions.values())[start:end]
        logging.debug(
            f"Displaying predictions {start} to {end} (total: {len(predictions_to_display)})"
        )
        for pred in predictions_to_display:
            if pred["predicted_outcome"] == pred["team_a"]:
                wager_odds = pred["team_a_odds"]
            elif pred["predicted_outcome"] == pred["team_b"]:
                wager_odds = pred["team_b_odds"]
            else:
                wager_odds = pred["tie_odds"] if "tie_odds" in pred else "N/A"

            if pred["outcome"] == "Wager Won":
                payout = pred["wager"] * wager_odds
            elif pred["outcome"] == "Wager Lost":
                payout = 0
            else:
                payout = "Pending"

            table.add_row(
                self.format_date(pred["prediction_date"]),
                pred["game_id"],
                pred["team_a"],
                pred["team_b"],
                pred["predicted_outcome"],
                f"${pred['wager']:.2f}",
                f"{wager_odds:.2f}"
                if isinstance(wager_odds, (int, float))
                else str(wager_odds),
                pred["outcome"],
                f"${payout:.2f}" if isinstance(payout, (int, float)) else str(payout),
                str(pred["validators_sent_to"]),
                str(pred["validators_confirmed"]),
            )

        if self.predictions_search_mode:
            footer = f"Search: {self.predictions_search_query} | Page {self.page} of {max(1, (len(predictions) - 1) // self.items_per_page + 1)}"
        else:
            footer = f"Page {self.page} of {max(1, (len(predictions) - 1) // self.items_per_page + 1)}"
        return Panel(
            Group(
                Text(f"Miner UID: {self.miner_uid}", style=f"bold {GOLD}"),
                table,
                Text(footer, justify="center", style=LIGHT_GREEN),
            ),
            title="Predictions",
            border_style=DARK_GREEN,
        )

    def generate_games_view(self):
        if self.games_handler is None:
            return Panel(
                Text(
                    "No games available. Please select a valid miner.",
                    style=LIGHT_GREEN,
                ),
                title="Games",
                border_style=DARK_GREEN,
            )

        games = self.games_handler.get_active_games()
        if self.search_query:
            games = {
                k: v
                for k, v in games.items()
                if self.search_query.lower() in v.sport.lower()
                or self.search_query.lower() in v.team_a.lower()
                or self.search_query.lower() in v.team_b.lower()
            }

        if not games:
            return Panel(
                Group(
                    self.generate_miner_info(),
                    Text(
                        "No games found matching the search criteria.",
                        justify="center",
                        style=LIGHT_GREEN,
                    ),
                ),
                title="Active Games",
                border_style=DARK_GREEN,
            )

        max_widths = {
            "Sport": max((len(game.sport) for game in games.values()), default=10),
            "Team A": max((len(game.team_a) for game in games.values()), default=10),
            "Team B": max((len(game.team_b) for game in games.values()), default=10),
            "Start Date": len("YYYY-MM-DD HH:MM"),
            "Team A Odds": len("100.00"),
            "Team B Odds": len("100.00"),
            "Tie Odds": len("100.00"),
        }

        table = Table(box=box.ROUNDED, expand=True, border_style=DARK_GREEN)
        table.add_column("Sport", style=LIGHT_GREEN, width=max_widths["Sport"])
        table.add_column("Team A", style=GOLD, width=max_widths["Team A"])
        table.add_column("Team B", style=GOLD, width=max_widths["Team B"])
        table.add_column(
            "Start Date", style=LIGHT_GREEN, width=max_widths["Start Date"]
        )
        table.add_column(
            "Team A Odds", style=LIGHT_GOLD, width=max_widths["Team A Odds"]
        )
        table.add_column(
            "Team B Odds", style=LIGHT_GOLD, width=max_widths["Team B Odds"]
        )
        table.add_column("Tie Odds", style=LIGHT_GOLD, width=max_widths["Tie Odds"])

        start = (self.page - 1) * self.items_per_page
        end = start + self.items_per_page
        for i, (game_id, game) in enumerate(list(games.items())[start:end]):
            style = "reverse" if i == self.cursor_position else ""
            table.add_row(
                game.sport,
                game.team_a,
                game.team_b,
                self.format_event_start_date(game.event_start_date),
                f"{game.team_a_odds:.2f}",
                f"{game.team_b_odds:.2f}",
                f"{game.tie_odds:.2f}" if game.tie_odds is not None else "N/A",
                style=style,
            )

        if self.search_mode:
            footer = f"Search: {self.search_query} | Page {self.page} of {max(1, (len(games) - 1) // self.items_per_page + 1)}"
        else:
            footer = f"Page {self.page} of {max(1, (len(games) - 1) // self.items_per_page + 1)}"

        return Panel(
            Group(
                self.generate_miner_info(),
                table,
                Text(footer, justify="center", style=LIGHT_GREEN),
            ),
            title="Active Games",
            border_style=DARK_GREEN,
        )

    def enter_prediction_for_selected_game(self):
        if not self.games_handler or not self.miner_uid:
            return Panel(
                Text(
                    "Unable to enter prediction. Please select a valid miner.",
                    style=LIGHT_GREEN,
                ),
                title="Enter Prediction",
                border_style=DARK_GREEN,
            )

        games = list(self.games_handler.get_active_games().items())
        start = (self.page - 1) * self.items_per_page
        selected_game_id, selected_game = games[start + self.cursor_position]

        self.current_view = "enter_prediction"
        self.selected_game = selected_game
        self.prediction_outcome = None
        self.prediction_wager = None
        self.wager_input = ""
        self.wager_input_mode = False
        self.last_prediction_time = 0
        self.last_prediction_id = None
        self.can_tie = selected_game.can_tie
        self.submission_message = None
        self.confirmation_mode = False

        self.reset_layout()
        get_app().invalidate()

    def generate_enter_prediction_view(self):
        if not self.games_handler or not self.miner_uid:
            return Panel(
                Text(
                    "Unable to enter prediction. Please select a valid miner.",
                    style=LIGHT_GREEN,
                ),
                title="Enter Prediction",
                border_style=DARK_GREEN,
            )

        if hasattr(self, "confirmation_mode") and self.confirmation_mode:
            content = [
                Text(self.submission_message, style="bold green", justify="center")
            ]
            return Panel(
                Group(*content),
                title="Prediction Confirmation",
                border_style=DARK_GREEN,
            )

        table = Table(show_header=False, box=box.ROUNDED, border_style=DARK_GREEN)
        table.add_column("Field", style=LIGHT_GREEN)
        table.add_column("Value", style=GOLD)

        table.add_row("Sport", self.selected_game.sport)
        table.add_row("Team A", self.selected_game.team_a)
        table.add_row("Team B", self.selected_game.team_b)
        table.add_row(
            "Start Date",
            self.format_event_start_date(self.selected_game.event_start_date),
        )
        table.add_row("Team A Odds", f"{self.selected_game.team_a_odds:.2f}")
        table.add_row("Team B Odds", f"{self.selected_game.team_b_odds:.2f}")
        table.add_row(
            "Tie Odds",
            f"{self.selected_game.tie_odds:.2f}"
            if self.selected_game.tie_odds is not None
            else "N/A",
        )
        table.add_row("Predicted Outcome", self.prediction_outcome or "Not selected")

        if self.wager_input_mode:
            table.add_row("Wager Amount", f"${self.wager_input}")
        else:
            table.add_row(
                "Wager Amount",
                f"${self.prediction_wager:.2f}"
                if self.prediction_wager is not None
                else "Not entered",
            )

        footer_text = "Press 'a' for Team A, 'b' for Team B, 't' for Tie, 'w' to enter/exit wager input, 'c' to confirm, 'esc' to cancel"
        if self.wager_input_mode:
            footer_text = f"Enter wager amount (press 'w' to confirm, backspace to delete). Current input: ${self.wager_input}"

        content = [self.generate_miner_info(), table]

        if hasattr(self, "submission_message") and self.submission_message:
            message_style = (
                "bold green" if "submitted" in self.submission_message else "bold red"
            )
            content.append(Text(self.submission_message, style=message_style))

        content.append(Text(footer_text, style=LIGHT_GREEN))

        return Panel(Group(*content), title="Enter Prediction", border_style=DARK_GREEN)

    def filter_games(self):
        sport = Prompt.ask("Enter sport to filter (or leave blank for all)")
        date = Prompt.ask("Enter date to filter (YYYY-MM-DD) (or leave blank for all)")

        games = self.games_handler.get_active_games()
        filtered_games = {}
        for game_id, game in games.items():
            if (not sport or game.sport.lower() == sport.lower()) and (
                not date or game.eventStartDate.startswith(date)
            ):
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
            for _ in range(30):
                live.update(Text("\n".join(rows), style=GOLD))
                rows = rows[1:] + [get_money_row()]
                time.sleep(0.05)

    def is_not_searching(self):
        return not (self.search_mode or self.predictions_search_mode)

    def setup_keybindings(self):
        @self.kb.add("q", filter=Condition(lambda: self.is_not_searching()))
        @self.kb.add(Keys.ControlC)
        @self.kb.add(Keys.ControlX)
        @self.kb.add(Keys.ControlZ)
        def _(event):
            self.quit()

        @self.kb.add("m", filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.current_view = "main_menu"
            self.page = 1
            self.cursor_position = 0

        @self.kb.add("p", filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.current_view = "predictions"
            self.page = 1
            self.cursor_position = 0

        @self.kb.add("g", filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.current_view = "games"
            self.page = 1
            self.cursor_position = 0

        @self.kb.add(
            "n",
            filter=Condition(lambda: self.is_not_searching() and self.available_miners),
        )
        def _(event):
            self.select_next_miner()

        @self.kb.add("f", filter=Condition(lambda: self.is_not_searching()))
        def _(event):
            self.filter_games()

        @self.kb.add(
            "s",
            filter=Condition(lambda: self.current_view == "games" and not self.search_mode)
        )
        def _(event):
            self.search_mode = True
            self.search_query = ""

        @self.kb.add(
            "s",
            filter=Condition(
                lambda: self.current_view == "predictions"
                and not self.predictions_search_mode
            ),
        )
        def _(event):
            self.predictions_search_mode = True
            self.predictions_search_query = ""

        @self.kb.add(
            Keys.Escape,
            filter=Condition(lambda: self.search_mode or self.predictions_search_mode),
        )
        def _(event):
            self.search_mode = False
            self.predictions_search_mode = False
            self.search_query = ""
            self.predictions_search_query = ""
            self.page = 1
            self.cursor_position = 0

        @self.kb.add(
            Keys.Any,
            filter=Condition(lambda: self.search_mode or self.predictions_search_mode),
        )
        def _(event):
            logging.debug(f"Key pressed in search mode: {repr(event.data)}")

            if event.data == Keys.ControlJ:
                logging.debug("Enter key pressed, exiting search mode")
                self.search_mode = False
                self.predictions_search_mode = False
            elif event.data in (
                Keys.Backspace,
                Keys.Delete,
                "\x7f",
            ):
                logging.debug(f"Backspace or Delete key pressed: {repr(event.data)}")
                if self.search_mode and self.search_query:
                    self.search_query = self.search_query[:-1]
                    logging.debug(f"Updated search query: {self.search_query}")
                elif self.predictions_search_mode and self.predictions_search_query:
                    self.predictions_search_query = self.predictions_search_query[:-1]
                    logging.debug(
                        f"Updated predictions search query: {self.predictions_search_query}"
                    )
            elif event.data and len(event.data) == 1 and event.data.isprintable():
                if self.search_mode:
                    self.search_query += event.data
                    logging.debug(
                        f"Added character to search query: {self.search_query}"
                    )
                else:
                    self.predictions_search_query += event.data
                    logging.debug(
                        f"Added character to predictions search query: {self.predictions_search_query}"
                    )
            else:
                logging.debug(f"Unrecognized key: {repr(event.data)}")

            self.page = 1
            self.cursor_position = 0
            event.app.invalidate()

        @self.kb.add(
            Keys.Backspace,
            filter=Condition(lambda: self.search_mode or self.predictions_search_mode),
        )
        def _(event):
            logging.debug("Explicit Backspace key pressed")
            if self.search_mode and self.search_query:
                self.search_query = self.search_query[:-1]
                logging.debug(f"Updated search query: {self.search_query}")
            elif self.predictions_search_mode and self.predictions_search_query:
                self.predictions_search_query = self.predictions_search_query[:-1]
                logging.debug(
                    f"Updated predictions search query: {self.predictions_search_query}"
                )
            self.page = 1
            self.cursor_position = 0
            event.app.invalidate()

        @self.kb.add(
            "up",
            filter=Condition(lambda: self.current_view in ["games", "predictions"]),
        )
        def _(event):
            self.cursor_position = max(0, self.cursor_position - 1)

        @self.kb.add(
            "down",
            filter=Condition(lambda: self.current_view in ["games", "predictions"]),
        )
        def _(event):
            self.cursor_position = min(
                self.items_per_page - 1, self.cursor_position + 1
            )

        @self.kb.add(
            "left",
            filter=Condition(lambda: self.current_view in ["games", "predictions"]),
        )
        def _(event):
            self.page = max(1, self.page - 1)
            self.cursor_position = 0

        @self.kb.add(
            "right",
            filter=Condition(lambda: self.current_view in ["games", "predictions"]),
        )
        def _(event):
            self.page += 1
            self.cursor_position = 0

        @self.kb.add("enter", filter=Condition(lambda: self.current_view == "games"))
        def _(event):
            self.enter_prediction_for_selected_game()

        @self.kb.add(
            "a", filter=Condition(lambda: self.current_view == "enter_prediction")
        )
        def _(event):
            self.prediction_outcome = self.selected_game.team_a

        @self.kb.add(
            "b", filter=Condition(lambda: self.current_view == "enter_prediction")
        )
        def _(event):
            self.prediction_outcome = self.selected_game.team_b

        @self.kb.add(
            "t",
            filter=Condition(
                lambda: self.current_view == "enter_prediction" and self.can_tie
            ),
        )
        def _(event):
            self.prediction_outcome = "Tie"

        @self.kb.add(
            "w", filter=Condition(lambda: self.current_view == "enter_prediction")
        )
        def _(event):
            if self.wager_input_mode:
                self.wager_input_mode = False
                try:
                    self.prediction_wager = float(self.wager_input)
                except ValueError:
                    self.prediction_wager = None
                finally:
                    self.wager_input = ""
            else:
                self.wager_input_mode = True
                self.wager_input = ""
            event.app.invalidate()

        @self.kb.add(Keys.Any, filter=Condition(lambda: self.wager_input_mode))
        def _(event):
            logging.debug(f"Key pressed in wager input mode: {event.data}")
            if event.data in (Keys.Backspace, Keys.Delete, "\x7f"):
                logging.debug("Backspace or Delete key pressed")
                self.wager_input = self.wager_input[:-1]
                logging.debug(f"Updated wager input: {self.wager_input}")
            elif len(event.data) == 1 and (event.data.isdigit() or event.data == "."):
                self.wager_input += event.data
                logging.debug(f"Added character to wager input: {self.wager_input}")
            event.app.invalidate()

        @self.kb.add(
            Keys.Escape,
            filter=Condition(lambda: self.current_view == "enter_prediction"),
        )
        def _(event):
            if self.wager_input_mode:
                self.wager_input_mode = False
                self.wager_input = ""
                self.prediction_wager = None
            else:
                self.current_view = "games"
            self.esc_pressed_once = False
            event.app.invalidate()

        @self.kb.add(
            "c",
            filter=Condition(
                lambda: self.current_view == "enter_prediction"
                and not self.wager_input_mode
            ),
        )
        def _(event):
            logging.info(
                f"'c' key pressed. Current view: {self.current_view}, Wager input mode: {self.wager_input_mode}"
            )
            logging.info(
                f"Prediction outcome: {self.prediction_outcome}, Prediction wager: {self.prediction_wager}"
            )
            if self.prediction_outcome and self.prediction_wager is not None:
                logging.info(
                    f"Submitting prediction: {self.prediction_outcome}, {self.prediction_wager}"
                )
                self.submit_prediction()
            else:
                logging.warning(
                    "Attempted to submit prediction without outcome or wager"
                )
            event.app.invalidate()

        @self.kb.add(
            Keys.Enter,
            filter=Condition(lambda: self.search_mode or self.predictions_search_mode),
        )
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

        @self.kb.add(
            Keys.Any,
            filter=Condition(
                lambda: hasattr(self, "confirmation_mode") and self.confirmation_mode
            ),
        )
        def _(event):
            pass

    def select_next_miner(self):
        if not self.available_miners:
            self.console_print("[bold red]No miners available to select.[/bold red]")
            return

        current_index = next(
            (
                i
                for i, miner in enumerate(self.available_miners)
                if str(miner["miner_uid"]) == str(self.miner_uid)
            ),
            -1,
        )
        next_index = (current_index + 1) % len(self.available_miners)
        next_miner = self.available_miners[next_index]
        self.miner_uid = str(next_miner["miner_uid"])
        self.miner_hotkey = str(next_miner["miner_hotkey"])
        self.miner_is_active = next_miner["is_active"]
        self.save_miner_uid(self.miner_uid)
        self.reload_miner_data()
        self.reset_layout()
        self.add_log_message(f"Switched to miner: {self.miner_uid}")

    def quit(self):
        if self.state_manager:
            self.state_manager.save_state()
        self.running = False
        get_app().exit()

    def get_saved_miner_uid(self):
        """
        Retrieve the saved miner UID from file.
        If the file doesn't exist, return None.
        """
        file_path = "current_miner_uid.txt"
        try:
            with open(file_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            bt.logging.warning(
                f"{file_path} not found. Will use the first available miner."
            )
            return None

    def save_miner_uid(self, uid):
        """
        Save the current miner UID to a file.
        """
        file_path = "current_miner_uid.txt"
        with open(file_path, "w") as f:
            f.write(str(uid))

    def run(self):
        try:
            if not self.available_miners:
                self.console_print(
                    "[bold red]No miners available. Please check your configuration.[/bold red]"
                )
                self.console_print(
                    "[bold yellow]The application will run with limited functionality.[/bold yellow]"
                )
                self.console_print(
                    "[bold yellow]You won't be able to switch miners or make predictions.[/bold yellow]"
                )

            layout = PromptLayout(
                Window(content=FormattedTextControl(self.get_formatted_text))
            )
            app = PromptApplication(
                layout=layout, key_bindings=self.kb, full_screen=True
            )

            def exit_handler(signum, frame):
                self.quit()

            signal.signal(signal.SIGINT, exit_handler)
            signal.signal(signal.SIGTERM, exit_handler)

            app.run()
        except Exception as e:
            self.console_print(f"[bold red]An error occurred: {e}[/bold red]")
            self.console_print(
                f"[bold yellow]Please check the log file: {log_file}[/bold yellow]"
            )
            logging.error(f"Unhandled exception: {e}", exc_info=True)
        finally:
            if self.state_manager:
                self.state_manager.save_state()

    def get_formatted_text(self):
        layout = self.generate_layout()
        console = Console(width=self.console.width, height=self.console.height)
        with console.capture() as capture:
            console.print(layout)
        logging.debug(f"Last key press: {self.last_key_press}")
        return ANSI(capture.get())

    def clear_console(self):
        # For Windows
        if os.name == "nt":
            _ = os.system("cls")
        # For Unix/Linux/MacOS
        else:
            _ = os.system("clear")

    def submit_prediction(self):
        current_time = time.time()
        if current_time - self.last_prediction_time < 1:  # 1 second cooldown
            self.submission_message = "Please wait before submitting another prediction."
            return

        if not self.prediction_outcome or not self.prediction_wager:
            self.submission_message = "Please select an outcome and enter a wager."
            return

        try:
            prediction = {
                "prediction_id": str(uuid.uuid4()),
                "miner_uid": self.miner_uid,
                "game_id": self.selected_game.game_id,
                "prediction_date": datetime.now(timezone.utc).isoformat(),
                "predicted_outcome": self.prediction_outcome,
                "team_a": self.selected_game.team_a,
                "team_b": self.selected_game.team_b,
                "wager": self.prediction_wager,
                "team_a_odds": self.selected_game.team_a_odds,
                "team_b_odds": self.selected_game.team_b_odds,
                "tie_odds": self.selected_game.tie_odds,
                "outcome": "Unfinished",
                "model_name": None,
                "confidence_score": None,
            }

            result = self.predictions_handler.add_prediction(prediction)
            if result["status"] == "success":
                self.miner_cash -= self.prediction_wager
                self.miner_lifetime_wager += self.prediction_wager
                
                self.db_manager.execute_query(
                    "UPDATE miner_stats SET miner_cash = %s, miner_lifetime_wager_amount = %s WHERE miner_uid = %s",
                    (self.miner_cash, self.miner_lifetime_wager, self.miner_uid)
                )
                
                self.reload_miner_data()
                self.submission_message = f"Prediction submitted: {self.prediction_outcome} with wager ${self.prediction_wager:.2f}"
                self.add_log_message(self.submission_message)
                self.last_prediction_time = current_time
                self.last_prediction_id = prediction["prediction_id"]
                self.confirmation_mode = True
                app = get_app()
                app.create_background_task(self.delayed_return_to_games())
            else:
                self.add_log_message(f"Failed to submit prediction: {result['message']}")
        except Exception as e:
            self.add_log_message(f"Error in submit_prediction: {str(e)}")

        self.reset_layout()
        get_app().invalidate()

    async def delayed_return_to_games(self):
        await asyncio.sleep(2)
        self.current_view = "games"
        self.submission_message = None
        self.confirmation_mode = False
        self.prediction_outcome = None
        self.prediction_wager = None
        self.wager_input = ""
        self.wager_input_mode = False
        self.reset_layout()
        get_app().invalidate()

    def generate_miner_info(self):
        current_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        miner_info = Table.grid(padding=(0, 1))
        miner_info.add_column(style="bold " + LIGHT_GREEN, justify="right")
        miner_info.add_column(style=GOLD, justify="left")
        miner_info.add_row("Miner UID:", self.miner_uid)
        miner_info.add_row("Cash:", f"${self.miner_cash:.2f}")
        miner_info.add_row("Lifetime Wager:", f"${self.miner_lifetime_wager:.2f}")
        miner_info.add_row(
            "Last Prediction:",
            self.format_last_prediction_date(self.miner_last_prediction_date),
        )

        is_active = self.db_manager.is_miner_active(self.miner_uid)
        active_status = "Active" if is_active else "Inactive"
        active_style = GOLD if is_active else "bold red"
        miner_info.add_row("Status:", Text(active_status, style=active_style))

        miner_info.add_row("Current Time:", current_time_utc)

        panel = Panel(
            miner_info,
            title="Miner Info",
            border_style=DARK_GREEN,
            width=60,
            expand=False,
        )

        centered_panel = Align.center(panel)

        return Group(centered_panel)

    def reset_layout(self):
        try:
            new_layout = self.generate_layout()
            if new_layout is not None:
                self.layout = new_layout
                if self.application:
                    self.application.layout = new_layout
            else:
                self.add_log_message("Error: Unable to generate layout")
                self.layout = Layout(Panel("Error generating layout. Please check logs."))
        except Exception as e:
            self.add_log_message(f"Error in reset_layout: {str(e)}")
            self.layout = Layout(Panel(f"Error generating layout: {str(e)}"))
        
        get_app().invalidate()

    def refresh_logs(self):
        if not hasattr(self, 'layout') or self.layout is None:
            return

        log_content = "\n".join(self.log_messages[-5:]) if self.log_messages else "No logs yet."
        log_panel = Panel(log_content, title="Logs", border_style="blue")

        if hasattr(self.layout, 'children'):
            logs_exists = any(child.name == 'logs' for child in self.layout.children if hasattr(child, 'name'))
            if logs_exists:
                for child in self.layout.children:
                    if hasattr(child, 'name') and child.name == 'logs':
                        child.update(log_panel)
                        break
        
        get_app().invalidate()

    def add_log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        self.log_messages.append(formatted_message)
        self.log_message_times.append(time.time())
        self.clean_old_messages()
        self.refresh_logs()

    def clean_old_messages(self):
        current_time = time.time()
        while self.log_message_times and current_time - self.log_message_times[0] > self.log_display_time:
            self.log_messages.pop(0)
            self.log_message_times.pop(0)

    def start_log_cleaner(self):
        def clean_logs_periodically():
            while True:
                time.sleep(1)
                self.clean_old_messages()
                self.refresh_logs()

        import threading
        threading.Thread(target=clean_logs_periodically, daemon=True).start()


if __name__ == "__main__":
    main()
