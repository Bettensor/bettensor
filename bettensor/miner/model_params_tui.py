import os
import psycopg2
from psycopg2.extras import RealDictCursor
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.application.current import get_app
from prompt_toolkit.keys import Keys
import bittensor as bt
import psycopg2.extras

DARK_GREEN = "dark_green"
LIGHT_GREEN = "green"
GOLD = "gold1"
LIGHT_GOLD = "yellow"


class ModelParamsTUI:
    def __init__(self, db_params):
        self.db_params = db_params
        self.console = Console()
        self.ensure_model_params_table_exists()
        self.miner_ids = self.get_all_miner_ids()
        self.current_miner_index = 0
        saved_miner_id = self.load_saved_miner_id()
        if self.miner_ids:
            if saved_miner_id and saved_miner_id in self.miner_ids:
                self.miner_id = saved_miner_id
            else:
                self.miner_id = self.miner_ids[0]
            self.load_params(self.miner_id)
        else:
            bt.logging.warning("No miner IDs found in the database.")
            self.miner_id = None
            self.params = {}
        self.kb = KeyBindings()
        self.edit_mode = False
        self.edit_value = ""
        self.error_message = ""
        self.explanations = {
            "soccer_model_on": "Toggle the soccer model on or off.",
            "wager_distribution_steepness": "Controls the steepness of the wager distribution.",
            "fuzzy_match_percentage": "Sets the percentage for fuzzy matching.",
            "minimum_wager_amount": "Sets the minimum wager amount (0-1000).",
            "max_wager_amount": "Sets the maximum wager amount (0-1000).",
            "top_n_games": "Sets the number of top games to consider.",
            "nfl_model_on": "Toggle the NFL model on or off.",
            "nfl_minimum_wager_amount": "Sets the minimum wager amount for NFL (0-1000).",
            "nfl_max_wager_amount": "Sets the maximum wager amount for NFL (0-1000).",
            "nfl_top_n_games": "Sets the number of most confident NFL games to consider.",
            "nfl_kelly_fraction_multiplier": "Sets the multiplier for the fractional kelly criterion.",
            "nfl_edge_threshold": "Sets the minimum betting edge threshold for NFL.",
            "nfl_max_bet_percentage": "Sets the maximum bet percentage of total bankroll for NFL.",
        }

        self.mode = "normal"
        self.setup_keybindings()
        self.cursor_position = 0
        self.update_view()

    def db_connect(self):
        return psycopg2.connect(**self.db_params)

    def execute_query(self, query, params=None, fetch_one=False):
        with self.db_connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch_one:
                    return cur.fetchone()
                return cur.fetchall()

    def ensure_model_params_table_exists(self):
        with self.db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                CREATE TABLE IF NOT EXISTS model_params (
                    id SERIAL PRIMARY KEY,
                    miner_uid TEXT UNIQUE NOT NULL,
                    soccer_model_on BOOLEAN,
                    wager_distribution_steepness INTEGER,
                    fuzzy_match_percentage INTEGER,
                    minimum_wager_amount FLOAT,
                    max_wager_amount FLOAT,
                    top_n_games INTEGER,
                    nfl_model_on BOOLEAN,
                    nfl_minimum_wager_amount FLOAT,
                    nfl_max_wager_amount FLOAT,
                    nfl_top_n_games INTEGER,
                    nfl_kelly_fraction_multiplier FLOAT,
                    nfl_edge_threshold FLOAT,
                    nfl_max_bet_percentage FLOAT
                )
                """
                )
            conn.commit()

    def get_all_miner_ids(self):
        with self.db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT miner_uid FROM model_params ORDER BY miner_uid")
                return [row[0] for row in cur.fetchall()]

    def load_params(self, miner_id):
        query = "SELECT * FROM model_params WHERE miner_uid = %s"
        params = self.execute_query(query, (miner_id,), fetch_one=True)
        if params:
            self.params = dict(params)
            self.miner_id = miner_id
        else:
            self.create_default_params()

    def save_params(self):
        with self.db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                UPDATE model_params SET
                    soccer_model_on = %s,
                    wager_distribution_steepness = %s,
                    fuzzy_match_percentage = %s,
                    minimum_wager_amount = %s,
                    max_wager_amount = %s,
                    top_n_games = %s,
                    nfl_model_on = %s,
                    nfl_minimum_wager_amount = %s,
                    nfl_max_wager_amount = %s,
                    nfl_top_n_games = %s,
                    nfl_kelly_fraction_multiplier = %s,
                    nfl_edge_threshold = %s,
                    nfl_max_bet_percentage = %s
                WHERE miner_uid = %s
                """,
                    (
                        self.params["soccer_model_on"],
                        self.params["wager_distribution_steepness"],
                        self.params["fuzzy_match_percentage"],
                        self.params["minimum_wager_amount"],
                        self.params["max_wager_amount"],
                        self.params["top_n_games"],
                        self.params["nfl_model_on"],
                        self.params["nfl_minimum_wager_amount"],
                        self.params["nfl_max_wager_amount"],
                        self.params["nfl_top_n_games"],
                        self.params["nfl_kelly_fraction_multiplier"],
                        self.params["nfl_edge_threshold"],
                        self.params["nfl_max_bet_percentage"],
                        self.miner_id,
                    ),
                )
            conn.commit()

    def setup_keybindings(self):
        @self.kb.add("c-c")
        @self.kb.add("c-x")
        @self.kb.add("c-z")
        def _(event):
            self.save_current_miner_id()
            event.app.exit()

        @self.kb.add("n", filter=Condition(lambda: not self.edit_mode))
        def _(event):
            self.select_next_miner()
            event.app.invalidate()

        @self.kb.add("up", filter=Condition(lambda: not self.edit_mode))
        def _(event):
            self.cursor_position = max(0, self.cursor_position - 1)
            self.update_view()

        @self.kb.add("down", filter=Condition(lambda: not self.edit_mode))
        def _(event):
            self.cursor_position = min(len(self.params) - 2, self.cursor_position + 1)
            self.update_view()

        @self.kb.add("enter")
        def _(event):
            self.toggle_edit_mode()

        @self.kb.add("escape")
        def _(event):
            if self.edit_mode:
                self.edit_mode = False
                self.edit_value = ""
                self.error_message = ""
                self.update_view()

        @self.kb.add("backspace")
        def _(event):
            if self.edit_mode:
                self.edit_value = self.edit_value[:-1]
                self.update_view()

        @self.kb.add("space")
        def _(event):
            if self.edit_mode:
                self.edit_value += " "
                self.update_view()

        for key in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-":
            @self.kb.add(key)
            def _(event, key=key):
                if self.edit_mode:
                    self.edit_value += key
                    self.update_view()

        @self.kb.add("n")
        def _(event):
            self.select_next_miner()
            event.app.invalidate()

    def toggle_edit_mode(self):
        if self.mode == "normal":
            self.mode = "edit"
            key = list(self.params.keys())[self.cursor_position + 1]
            if key in ["soccer_model_on", "nfl_model_on"]:
                self.params[key] = not self.params[key]
                self.save_params()
                self.update_view()
            else:
                self.edit_mode = True
                self.edit_value = str(self.params[key])
        else:
            self.mode = "normal"
            self.edit_mode = False
            key = list(self.params.keys())[self.cursor_position + 1]
            new_value = self.validate_input(key, self.edit_value)
            if new_value is not None:
                self.params[key] = new_value
                self.save_params()
                self.error_message = ""
            else:
                self.error_message = f"Invalid input for {key}. Please try again."
            self.edit_value = ""
        self.update_view()

    def validate_input(self, key, value):
        if key in ["soccer_model_on", "nfl_model_on"]:
            return value.lower() in ['true', '1', 'yes', 'on']
        try:
            if key == "wager_distribution_steepness":
                int_value = int(value)
                if int_value > 0:
                    return int_value
                else:
                    return None
            elif key in [
                "fuzzy_match_percentage",
                "top_n_games",
                "nfl_top_n_games",
            ]:
                return int(value)
            else:
                return float(value)
        except ValueError:
            return None

    def create_table(self, width):
        table = Table(
            title=f"Model Parameters for Miner ID: {self.miner_id}",
            box=box.ROUNDED,
            title_style=LIGHT_GOLD,
            width=width,
        )
        table.add_column("Parameter", style=DARK_GREEN, width=int(width * 0.4))
        table.add_column("Value", style=LIGHT_GREEN, width=int(width * 0.6))

        for i, (key, value) in enumerate(self.params.items()):
            if key not in ["id", "miner_uid"]:
                style = f"reverse {LIGHT_GREEN}" if i - 1 == self.cursor_position else LIGHT_GREEN
                if self.edit_mode and i - 1 == self.cursor_position:
                    value = f"{self.edit_value}▋"
                elif key in ["soccer_model_on", "nfl_model_on"]:
                    value = "On" if value else "Off"
                table.add_row(key.replace("_", " ").title(), str(value), style=style)

        return table

    def create_explanation(self, width):
        current_param = list(self.params.keys())[self.cursor_position + 1]
        explanation = self.explanations.get(current_param, "No explanation available.")
        if self.error_message:
            explanation = f"[red]{self.error_message}[/red]\n{explanation}"
        return Panel(
            explanation,
            title="Parameter Explanation",
            width=width,
            border_style=LIGHT_GREEN,
        )

    def create_legend(self, width):
        legend = Table(
            title="Key Bindings", box=box.ROUNDED, title_style=LIGHT_GOLD, width=width
        )
        legend.add_column("Key", style=DARK_GREEN, width=int(width * 0.3))
        legend.add_column("Action", style=LIGHT_GREEN, width=int(width * 0.7))
        legend.add_row("↑/↓", "Navigate")
        legend.add_row("Enter", "Edit/Save")
        legend.add_row("Esc", "Cancel Edit")
        legend.add_row("n", "Next Miner")
        legend.add_row("^C/^X/^Z", "Quit")
        return legend

    def get_formatted_text(self):
        app = get_app()
        width, height = os.get_terminal_size()
        width = min(width, 100)  # Cap the width at 100 to prevent overly wide tables

        table = self.create_table(width)
        explanation = self.create_explanation(width)
        legend = self.create_legend(width)

        legend_height = 8
        explanation_height = 3
        available_table_height = height - legend_height - explanation_height - 6

        with self.console.capture() as capture:
            self.console.print(table, height=available_table_height)
            self.console.print("\n")
            self.console.print(explanation)
            self.console.print("\n")
            self.console.print(legend)

        return ANSI(capture.get())

    def run(self):
        if not self.miner_ids:
            print("No miner parameters found in the database.")
            return

        layout = Layout(Window(content=FormattedTextControl(self.get_formatted_text)))
        self.application = Application(
            layout=layout,
            key_bindings=self.kb,
            full_screen=True,
        )
        bt.logging.info("Starting application")
        try:
            self.application.run()
        except Exception as e:
            bt.logging.error(f"Error running application: {str(e)}")

    def update_view(self):
        if hasattr(self, "application"):
            new_content = self.get_formatted_text()
            self.application.layout.container.content = FormattedTextControl(new_content)
            self.application.invalidate()

    def select_next_miner(self):
        if not self.miner_ids:
            return

        current_index = self.miner_ids.index(self.miner_id)
        next_index = (current_index + 1) % len(self.miner_ids)
        self.miner_id = self.miner_ids[next_index]
        self.load_params(self.miner_id)
        self.cursor_position = 0
        self.update_view()
        if hasattr(self, 'application'):
            self.application.invalidate()
        self.save_current_miner_id()

    def create_default_params(self):
        default_params = {
            "miner_uid": self.miner_id,
            "soccer_model_on": False,
            "wager_distribution_steepness": 1,
            "fuzzy_match_percentage": 80,
            "minimum_wager_amount": 1.0,
            "max_wager_amount": 100.0,
            "top_n_games": 10,
            "nfl_model_on": False,
            "nfl_minimum_wager_amount": 1.0,
            "nfl_max_wager_amount": 100.0,
            "nfl_top_n_games": 5,
            "nfl_kelly_fraction_multiplier": 1.0,
            "nfl_edge_threshold": 0.02,
            "nfl_max_bet_percentage": 0.7,
        }
        with self.db_connect() as conn:
            with conn.cursor() as cur:
                columns = ', '.join(default_params.keys())
                values = ', '.join(['%s'] * len(default_params))
                query = f"INSERT INTO model_params ({columns}) VALUES ({values})"
                cur.execute(query, list(default_params.values()))
            conn.commit()
        self.params = default_params

    def save_current_miner_id(self):
        file_path = "current_miner_uid.txt"
        with open(file_path, "w") as f:
            f.write(str(self.miner_id))

    def load_saved_miner_id(self):
        file_path = "current_miner_uid.txt"
        try:
            with open(file_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None


if __name__ == "__main__":
    DB_NAME = os.getenv("DB_NAME", "bettensor")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "bettensor_password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    db_params = {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
    }

    tui = ModelParamsTUI(db_params)
    tui.run()