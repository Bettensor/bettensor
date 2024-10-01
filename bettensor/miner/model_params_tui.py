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

# Define custom colors
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
        self.load_params()
        self.kb = KeyBindings()
        self.edit_mode = False
        self.edit_value = ""
        self.error_message = ""
        self.explanations = {
            "model_on": "Toggle the soccer model on or off.",
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

        self.setup_keybindings()
        self.cursor_position = 0
        self.update_view()

    def db_connect(self):
        return psycopg2.connect(**self.db_params)

    def ensure_model_params_table_exists(self):
        with self.db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                CREATE TABLE IF NOT EXISTS model_params (
                    id SERIAL PRIMARY KEY,
                    model_on BOOLEAN,
                    wager_distribution_steepness INTEGER,
                    fuzzy_match_percentage INTEGER,
                    minimum_wager_amount FLOAT,
                    max_wager_amount FLOAT,
                    top_n_games INTEGER
                )
                """
                )
            conn.commit()

    def get_all_miner_ids(self):
        with self.db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT miner_uid FROM model_params ORDER BY miner_uid")
                return [row[0] for row in cur.fetchall()]

    def load_params(self):
        if not self.miner_ids:
            self.params = None
            return

        self.miner_id = self.miner_ids[self.current_miner_index]
        with self.db_connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM model_params WHERE miner_uid = %s", (self.miner_id,)
                )
                self.params = cur.fetchone()

    def save_params(self):
        with self.db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                UPDATE model_params SET
                    model_on = %s,
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
                        self.params["model_on"],
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
            event.app.exit()

        @self.kb.add("n", filter=Condition(lambda: not self.edit_mode))
        def _(event):
            self.select_next_miner()

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

        # Add key bindings for editing
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

        # Add key bindings for all other characters
        for key in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-":

            @self.kb.add(key)
            def _(event, key=key):
                if self.edit_mode:
                    self.edit_value += key
                    self.update_view()

    def toggle_edit_mode(self):
        key = list(self.params.keys())[self.cursor_position + 1]
        if key in ["model_on", "nfl_model_on"]:
            self.params[key] = not self.params[key]
            self.save_params()
            self.update_view()
        elif not self.edit_mode:
            self.edit_mode = True
            self.edit_value = str(self.params[key])
        else:
            self.edit_mode = False
            try:
                new_value = self.validate_input(key, self.edit_value)
                if new_value is not None:
                    self.params[key] = new_value
                    self.save_params()
                    self.error_message = ""
                else:
                    self.error_message = f"Invalid input for {key}. Please try again."
            except ValueError:
                self.error_message = f"Invalid input for {key}. Please enter a number."
            self.edit_value = ""
        self.update_view()

    def validate_input(self, key, value):
        try:
            if key in [
                "wager_distribution_steepness",
                "fuzzy_match_percentage",
                "top_n_games",
            ]:
                return int(value)
            elif key in ["minimum_wager_amount", "max_wager_amount"]:
                float_value = float(value)
                if 0 <= float_value <= 1000:
                    return float_value
                else:
                    self.error_message = f"{key} must be between 0 and 1000."
                    return None
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
            if key != "id":
                style = (
                    f"reverse {LIGHT_GREEN}"
                    if i - 1 == self.cursor_position
                    else LIGHT_GREEN
                )
                if self.edit_mode and i - 1 == self.cursor_position:
                    value = f"{self.edit_value}▋"
                elif key in ["model_on", "nfl_model_on"]:
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

        # Calculate available heights
        legend_height = 8  # Approximate height of the legend table
        explanation_height = 3  # Height of the explanation panel
        available_table_height = (
            height - legend_height - explanation_height - 6
        )  # Subtract some padding

        with self.console.capture() as capture:
            self.console.print(table, height=available_table_height)
            self.console.print("\n")
            self.console.print(explanation)
            self.console.print("\n")
            self.console.print(legend)

        return ANSI(capture.get())

    def run(self):
        if not self.params:
            print("No miner parameters found in the database.")
            return

        layout = Layout(Window(content=FormattedTextControl(self.get_formatted_text())))

        self.application = Application(
            layout=layout,
            key_bindings=self.kb,
            full_screen=True,
        )
        self.application.run()

    def update_view(self):
        if hasattr(self, "application"):
            new_content = self.get_formatted_text()
            self.application.layout.container.content = FormattedTextControl(
                new_content
            )
            self.application.invalidate()

    def select_next_miner(self):
        self.current_miner_index = (self.current_miner_index + 1) % len(self.miner_ids)
        self.load_params()
        self.cursor_position = 0
        self.update_view()


if __name__ == "__main__":
    # Load environment variables
    DB_NAME = os.getenv("DB_NAME", "bettensor")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "bettensor_password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    # Database connection parameters
    db_params = {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
    }

    # Create and run ModelParamsTUI
    tui = ModelParamsTUI(db_params)
    tui.run()
