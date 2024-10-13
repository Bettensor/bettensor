import redis
import json
import os
import subprocess
import requests
import bittensor as bt
from bettensor.miner.interfaces.redis_interface import RedisInterface
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout import Layout as PromptLayout
from prompt_toolkit.formatted_text import ANSI

# Set up logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Define custom colors
DARK_GREEN = "dark_green"
LIGHT_GREEN = "green"
GOLD = "gold1"
LIGHT_GOLD = "yellow"


class MenuApplication:
    def __init__(self):
        self.console = Console()
        self.r = get_redis_client()
        self.kb = KeyBindings()
        self.setup_keybindings()

    def setup_keybindings(self):
        @self.kb.add("q")
        def _(event):
            event.app.exit()

        @self.kb.add("1")
        def _(event):
            event.app.exit()
            self.selected_option = self.access_cli

        @self.kb.add("2")
        def _(event):
            event.app.exit()
            self.selected_option = self.edit_model_parameters

        @self.kb.add("3")
        def _(event):
            event.app.exit()
            self.selected_option = self.sign_new_token

        @self.kb.add("4")
        def _(event):
            event.app.exit()
            self.selected_option = self.revoke_current_token

        @self.kb.add("5")
        def _(event):
            event.app.exit()
            self.selected_option = self.check_token_status

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def get_formatted_text(self):
        layout = self.generate_layout()
        with self.console.capture() as capture:
            self.console.print(layout)
        return ANSI(capture.get())

    def generate_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["header"].update(Panel("BetTensor Miner Menu", style=f"bold {GOLD}"))
        layout["body"].update(self.generate_menu())
        layout["footer"].update(Panel("Press q to quit", style=f"italic {LIGHT_GREEN}"))

        return layout

    def generate_menu(self):
        table = Table(show_header=False, box=None, expand=True, border_style=DARK_GREEN)
        table.add_column("Option", style=LIGHT_GREEN)
        table.add_column("Description", style=GOLD)

        options = [
            ("1", "Access CLI"),
            ("2", "Edit Model Parameters"),
            ("3", "Sign new token"),
            ("4", "Revoke current token"),
            ("5", "Check token status"),
        ]

        for option, description in options:
            table.add_row(f"[{option}]", description)

        return Panel(table, title="Main Menu", border_style=DARK_GREEN)

    def run(self):
        self.selected_option = None
        layout = PromptLayout(
            Window(content=FormattedTextControl(self.get_formatted_text))
        )
        app = Application(layout=layout, key_bindings=self.kb, full_screen=True)
        app.run()

        if self.selected_option:
            self.clear_screen()
            self.selected_option()

    def access_cli(self):
        subprocess.run(["python", "bettensor/miner/cli.py"])

    def edit_model_parameters(self):
        subprocess.run(["python", "bettensor/miner/model_params_tui.py"])

    def sign_new_token(self):
        from bettensor.miner.utils.sign_token import main as sign_token_main

        sign_token_main()

        print("\nToken signing process completed.")
        input("Press Enter to return to the main menu...")

    def revoke_current_token(self):
        token_data = get_stored_token()
        if token_data:
            self.r.publish(
                "token_management", json.dumps({"action": "revoke", "data": token_data})
            )
            response = self.r.blpop("token_management_response", timeout=5)
            if response:
                self.console.print(
                    json.loads(response[1])["message"], style=LIGHT_GREEN
                )
            else:
                self.console.print("No response from server.", style="red")
        else:
            self.console.print("No token found.", style="yellow")
        input("Press Enter to continue...")

    def check_token_status(self):
        token_data = get_stored_token()
        if token_data:
            self.r.publish(
                "token_management", json.dumps({"action": "check", "data": token_data})
            )
            response = self.r.blpop("token_management_response", timeout=5)
            if response:
                self.console.print(
                    json.loads(response[1])["message"], style=LIGHT_GREEN
                )
            else:
                self.console.print("No response from server.", style="red")
        else:
            self.console.print("No token found.", style="yellow")
        input("Press Enter to continue...")


def get_redis_client():
    return RedisInterface(host=REDIS_HOST, port=REDIS_PORT)


def get_stored_token():
    if os.path.exists("token_store.json"):
        with open("token_store.json", "r") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    while True:
        menu_app = MenuApplication()
        menu_app.run()
        if menu_app.selected_option is None:
            break
        menu_app.clear_screen()
