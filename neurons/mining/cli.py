# The MIT License (MIT)
# Copyright © 2024 geardici

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import bittensor as bt
import rich
import prompt_toolkit
from rich.console import Console
from rich.table import Table
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import Frame, TextArea, Label
import time  # Import time module for handling timeout
import threading  # Import threading for non-blocking delay
from prompt_toolkit.layout.containers import Window, HSplit

# TODO: add miner stat recalc method in miner.py
miner_stats = {
    "miner_hotkey": "1234567890",
    "miner_coldkey": "1234567890",
    "miner_cash": 750,
    "miner_status": "active",
    "miner_last_wager": "2024-06-08 12:00:00",
    "miner_lifetime_earnings": 2756,
    "miner_lifetime_wins": 6,
    "miner_lifetime_losses": 3,
    "miner_win_loss_ratio": 0.66666666666666666,
    "miner_rank": 1,
    "last_reward": 1.5
}

predictions = {
    "1001": {
        "sport": "Football", 
        "teamA": "Real Madrid", 
        "teamB": "FC Barcelona", 
        "startTime": "2024-06-07 12:00:00", 
        "predictionTime": "2024-06-07 11:00:00", 
        "odds": "1.5:2.5", 
        "prediction": "Real Madrid", 
        "wager_amount": 100,
        "can_overwrite": False
    },
    "1002": {
        "sport": "Basketball", 
        "teamA": "Lakers", 
        "teamB": "Clippers", 
        "startTime": "2024-06-09 12:00:00", 
        "predictionTime": "2024-06-08 11:00:00", 
        "odds": "1.8:1.9", 
        "prediction": "Clippers", 
        "wager_amount": 150,
        "can_overwrite": True
    }
}
gameData = {
    "1001": {"sport": "Football", "teamA": "Real Madrid", "teamB": "FC Barcelona", "startTime": "2024-06-07 12:00:00", "odds": "1.5:2.5"},
    "1002": {"sport": "Basketball", "teamA": "Lakers", "teamB": "Clippers", "startTime": "2024-06-09 12:00:00", "odds": "1.8:1.9"},
    "1003": {"sport": "Soccer", "teamA": "Manchester United", "teamB": "Chelsea", "startTime": "2024-06-10 12:00:00", "odds": "2.0:1.7"},
    "1004": {"sport": "Baseball", "teamA": "Yankees", "teamB": "Mets", "startTime": "2024-06-11 12:00:00", "odds": "1.6:2.2"},
    "1005": {"sport": "Hockey", "teamA": "Bruins", "teamB": "Canucks", "startTime": "2024-06-12 12:00:00", "odds": "1.9:2.0"},
    "1006": {"sport": "Football", "teamA": "Real Madrid", "teamB": "FC Barcelona", "startTime": "2024-06-13 12:00:00", "odds": "1.5:2.5"}
}


class InteractiveTable:
    '''
    Base class for interactive tables 
    '''
    def __init__(self, app):
        self.app = app
        self.text_area = TextArea(
            focusable=True,
            read_only=True,
            width=prompt_toolkit.layout.Dimension(preferred=70),
            height=prompt_toolkit.layout.Dimension(preferred=20)  # Use Dimension for height
        )
        self.frame = Frame(self.text_area, style="class:frame")
        self.box = HSplit([self.frame])  # Use HSplit to wrap Frame
        self.selected_index = 0
        self.options = []

    def update_text_area(self):
        lines = [f"> {option}" if i == self.selected_index else f"  {option}" for i, option in enumerate(self.options)]
        self.text_area.text = "\n".join(lines)

    def handle_enter(self):
        pass  # To be overridden

    def move_up(self):
        if self.selected_index > 0:
            self.selected_index -= 1
        self.update_text_area()

    def move_down(self):
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
        self.update_text_area()





class MainMenu(InteractiveTable):
    '''
    Main menu for the miner CLI - 1st level menu

    '''
    def __init__(self, app):
        super().__init__(app)
        self.miner_cash = app.miner_cash
        self.header = Label(" BetTensor Miner Main Menu", style="bold")  # Added leading space
        self.options = ["View/Edit Predictions", "View Games and Make Predictions"]
        self.update_text_area()

    def update_text_area(self):
        header_text = self.header.text
        divider = "-" * len(header_text)
        
        # Miner stats
        miner_stats_text = (
            f" Miner Hotkey: {miner_stats['miner_hotkey']}\n"
            f" Miner Coldkey: {miner_stats['miner_coldkey']}\n"
            f" Miner Cash: {self.miner_cash}\n" # TODO: Add second value for Actual vs Unsubmitted
            f" Miner Status: {miner_stats['miner_status']}\n"
            f" Last Wager: {miner_stats['miner_last_wager']}\n"
            f" Lifetime Earnings: ${miner_stats['miner_lifetime_earnings']}\n"
            f" Lifetime Wins: {miner_stats['miner_lifetime_wins']}\n"
            f" Lifetime Losses: {miner_stats['miner_lifetime_losses']}\n"
            f" Win/Loss Ratio: {miner_stats['miner_win_loss_ratio']:.2f}\n"
            f" Miner Rank: {miner_stats['miner_rank']}\n"
            f" Last Reward: {miner_stats['last_reward']} τ"
        )
        
        options_text = "\n".join(
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options)
        )
        
        self.text_area.text = f"{header_text}\n{divider}\n{miner_stats_text}\n{divider}\n{options_text}"

    def handle_enter(self):
        if self.selected_index == 0:
            self.app.change_view(PredictionsList(self.app))
        elif self.selected_index == 1:
            self.app.change_view(GamesList(self.app))

    def move_up(self):
        super().move_up()
        self.update_text_area()

    def move_down(self):
        super().move_down()
        self.update_text_area()


class PredictionsList(InteractiveTable):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.message = ""
        # Calculate maximum widths for each column
        max_sport_len = max(len(pred['sport']) for pred in predictions.values())
        max_teamA_len = max(len(pred['teamA']) for pred in predictions.values())
        max_teamB_len = max(len(pred['teamB']) for pred in predictions.values())
        max_startTime_len = max(len(pred['startTime']) for pred in predictions.values())
        max_odds_len = max(len(pred['odds']) for pred in predictions.values())
        max_prediction_len = max(len(pred['prediction']) for pred in predictions.values())
        max_status_len = max(len('Overwritable') if pred['can_overwrite'] else len('Final') for pred in predictions.values())

        # Define the header with calculated widths
        self.header = Label(
            f"  {'Sport':<{max_sport_len}} | {'Team A':<{max_teamA_len}} vs {'Team B':<{max_teamB_len}} | {'Start Time':<{max_startTime_len}} | {'Odds':<{max_odds_len}} | {'Prediction':<{max_prediction_len}} | {'Status':<{max_status_len}} ",
            style="bold"
        )

        # Generate options for the scrollable list
        self.options = [
            f"{pred['sport']:<{max_sport_len}} | {pred['teamA']:<{max_teamA_len}} vs {pred['teamB']:<{max_teamB_len}} | {pred['startTime']:<{max_startTime_len}} | {pred['odds']:<{max_odds_len}} | {pred['prediction']:<{max_prediction_len}} | {'Overwritable' if pred['can_overwrite'] else 'Final':<{max_status_len}}"
            for pred in predictions.values()
        ]
        self.options.append("Unsubmitted Predictions:")
        self.options.extend([
            f"{pred['sport']:<{max_sport_len}} | {pred['teamA']:<{max_teamA_len}} vs {pred['teamB']:<{max_teamB_len}} | {pred['startTime']:<{max_startTime_len}} | {pred['odds']:<{max_odds_len}} | {pred['prediction']:<{max_prediction_len}} | {'Overwritable' if pred['can_overwrite'] else 'Final':<{max_status_len}}"
            for pred in self.app.unsubmitted_predictions.values()
        ])
        self.options.append("Go Back")  # Add "Go Back" option
        self.update_text_area()

    def update_text_area(self):
        # Update the text area to include the header, divider, options, and space before "Go Back"
        header_text = self.header.text
        divider = "-" * len(header_text)
        options_text = "\n".join(
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options[:-1])
        )
        go_back_text = f"\n\n  {self.options[-1]}" if self.selected_index != len(self.options) - 1 else f"\n\n> {self.options[-1]}"
        self.text_area.text = f"{header_text}\n{divider}\n{options_text}{go_back_text}\n\n{self.message}"

    def handle_enter(self):
        if self.selected_index == len(self.options) - 1:  # Go Back
            self.app.change_view(MainMenu(self.app))
        elif self.selected_index == len(predictions):  # Unsubmitted Predictions label
            return  # Do nothing
        else:
            selected_index = self.selected_index
            if selected_index > len(predictions):
                selected_prediction = list(self.app.unsubmitted_predictions.values())[selected_index - len(predictions) - 1]
            else:
                selected_prediction = list(predictions.values())[selected_index]
            if selected_prediction['can_overwrite']:
                self.app.change_view(WagerConfirm(self.app, selected_prediction, self))
            else:
                self.message = "Too late to overwrite this prediction"
                self.update_text_area()
                threading.Timer(2.0, self.clear_message).start()

    def clear_message(self):
        self.message = ""
        self.update_text_area()

    def move_up(self):
        if self.selected_index > 0:
            self.selected_index -= 1
            if self.selected_index == len(predictions):  # Skip "Unsubmitted Predictions" label
                self.selected_index -= 1
            self.update_text_area()

    def move_down(self):
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
            if self.selected_index == len(predictions):  # Skip "Unsubmitted Predictions" label
                self.selected_index += 1
            self.update_text_area()

class GameResult(InteractiveTable):
    def __init__(self, app, prediction_data):
        super().__init__(app)
        self.prediction_data = prediction_data
        self.options = ["Go Back"]
        self.update_text_area()

    def update_text_area(self):
        prediction_info = f" {self.prediction_data['sport']} | {self.prediction_data['teamA']} vs {self.prediction_data['teamB']} | {self.prediction_data['startTime']} | {self.prediction_data['odds']} | Prediction: {self.prediction_data['prediction']}"  # Added leading space
        options_text = "\n".join(
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options)
        )
        self.text_area.text = f"{prediction_info}\n{options_text}"

    def handle_enter(self):
        if self.selected_index == 0:  # Go Back
            self.app.change_view(PredictionsList(self.app))

class GamesList(InteractiveTable):
    def __init__(self, app):
        super().__init__(app)
        # Calculate maximum widths for each column
        max_sport_len = max(len(game['sport']) for game in gameData.values())
        max_teamA_len = max(len(game['teamA']) for game in gameData.values())
        max_teamB_len = max(len(game['teamB']) for game in gameData.values())
        max_startTime_len = max(len(game['startTime']) for game in gameData.values())
        max_odds_len = max(len(game['odds']) for game in gameData.values())

        # Define the header with calculated widths
        self.header = Label(
            f"  {'Sport':<{max_sport_len}} | {'Team A':<{max_teamA_len}} vs {'Team B':<{max_teamB_len}} | {'Start Time':<{max_startTime_len}} | {'Odds':<{max_odds_len}} ",
            style="bold"
        )

        # Generate options for the scrollable list
        self.options = [
            f"{game['sport']:<{max_sport_len}} | {game['teamA']:<{max_teamA_len}} vs {game['teamB']:<{max_teamB_len}} | {game['startTime']:<{max_startTime_len}} | {game['odds']:<{max_odds_len}}"
            for game in gameData.values()
        ]
        self.options.append("Go Back")  # Add "Go Back" option
        self.update_text_area()

    def update_text_area(self):
        # Update the text area to include the header, divider, options, and space before "Go Back"
        header_text = self.header.text
        divider = "-" * len(header_text)
        options_text = "\n".join(
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options[:-1])
        )
        go_back_text = f"\n\n  {self.options[-1]}" if self.selected_index != len(self.options) - 1 else f"\n\n> {self.options[-1]}"
        self.text_area.text = f"{header_text}\n{divider}\n{options_text}{go_back_text}"

    def handle_enter(self):
        if self.selected_index == len(self.options) - 1:  # Go Back
            self.app.change_view(MainMenu(self.app))
        else:
            # Get the currently selected game data
            selected_game_data = gameData[list(gameData.keys())[self.selected_index]]
            # Change view to WagerConfirm, passing the selected game data
            self.app.change_view(WagerConfirm(self.app, selected_game_data, self))

    def move_up(self):
        if self.selected_index > 0:
            self.selected_index -= 1
            self.update_text_area()

    def move_down(self):
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
            self.update_text_area()



class WagerConfirm(InteractiveTable):
    '''
    Wager confirmation view
    '''
    def __init__(self, app, game_data, previous_view):
        super().__init__(app)
        self.game_data = game_data
        self.previous_view = previous_view
        self.miner_cash = app.miner_cash
        self.wager_input = TextArea(
            multiline=False,
            password=False,
            focusable=True
        )
        self.options = ["Enter Wager Amount", "Confirm Wager", "Go Back"]
        self.confirmation_message = ""
        self.update_text_area()

    def update_text_area(self):
        game_info = f" {self.game_data['sport']} | {self.game_data['teamA']} vs {self.game_data['teamB']} | {self.game_data['startTime']} | {self.game_data['odds']}"  # Added leading space
        cash_info = f"Miner's Cash: ${self.miner_cash}"
        wager_input_text = f"Wager Amount: {self.wager_input.text}"
        options_text = "\n".join(
            f"> {option}" if i == self.selected_index else f"  {option}"
            for i, option in enumerate(self.options)
        )
        self.text_area.text = f"{game_info}\n{cash_info}\n{wager_input_text}\n{options_text}\n\n{self.confirmation_message}"
        self.box = HSplit([self.text_area, self.wager_input])

    def handle_enter(self):
        if self.selected_index == 0:  # Enter Wager Amount
            self.focus_wager_input()
        elif self.selected_index == 1:  # Confirm Wager
            try:
                wager_amount = float(self.wager_input.text.strip())
                if wager_amount <= 0:
                    raise ValueError("Wager amount must be positive.")
                if wager_amount > self.miner_cash:
                    raise ValueError("Wager amount exceeds available cash.")
                # Add the prediction to unsubmitted_predictions
                prediction_id = str(len(self.app.unsubmitted_predictions) + 1)
                self.app.unsubmitted_predictions[prediction_id] = {
                    "sport": self.game_data['sport'],
                    "teamA": self.game_data['teamA'],
                    "teamB": self.game_data['teamB'],
                    "startTime": self.game_data['startTime'],
                    "predictionTime": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "odds": self.game_data['odds'],
                    "prediction": self.game_data['teamA'],  # Assuming prediction is for teamA
                    "wager_amount": wager_amount,
                    "can_overwrite": True
                }
                self.app.miner_cash -= wager_amount  # Deduct wager amount from miner's cash
                self.confirmation_message = "Wager confirmed! Returning to previous menu..."
                self.update_text_area()
                threading.Timer(2.0, lambda: self.app.change_view(self.previous_view)).start()  # Delay for 2 seconds
            except ValueError as e:
                self.wager_input.text = ""  # Reset the input text to an empty string
                self.confirmation_message = str(e)
                self.update_text_area()
                threading.Timer(2.0, lambda: self.app.change_view(self.previous_view)).start()  # Show error message for 2 seconds then return
        elif self.selected_index == 2:  # Go Back
            self.app.change_view(self.previous_view)

    def move_up(self):
        if self.selected_index > 0:
            self.selected_index -= 1
            self.update_text_area()

    def move_down(self):
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1
            self.update_text_area()

    def focus_wager_input(self):
        self.app.layout.focus(self.wager_input)

    def blur_wager_input(self):
        self.app.layout.focus(self.text_area)

    def handle_wager_input_enter(self):
        self.blur_wager_input()
        self.selected_index = 1  # Move focus to "Confirm Wager"
        self.update_text_area()
        self.app.layout.focus(self.text_area)  # Ensure focus is back on the text area


bindings = KeyBindings()

@bindings.add('up')
def _(event):
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, 'current_view'):
            custom_app.current_view.move_up()
            if isinstance(custom_app.current_view, WagerConfirm):
                custom_app.current_view.blur_wager_input()
    except AttributeError as e:
        print("Failed to move up:", str(e))

@bindings.add('down')
def _(event):
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, 'current_view'):
            custom_app.current_view.move_down()
            if isinstance(custom_app.current_view, WagerConfirm):
                custom_app.current_view.blur_wager_input()
    except AttributeError as e:
        print("Failed to move down:", str(e))

@bindings.add('enter')
def _(event):
    try:
        custom_app = event.app.custom_app
        if hasattr(custom_app, 'current_view'):
            if isinstance(custom_app.current_view, WagerConfirm):
                if custom_app.current_view.app.layout.has_focus(custom_app.current_view.wager_input):
                    custom_app.current_view.handle_wager_input_enter()
                else:
                    custom_app.current_view.handle_enter()
            else:
                custom_app.current_view.handle_enter()
    except AttributeError as e:
        print("Failed to handle enter:", str(e))

@bindings.add('q')
def _(event):
    event.app.exit()
    

class Application:
    def __init__(self, predictions, games, miner_stats):
        self.predictions = predictions
        self.unsubmitted_predictions = {}
        self.games = games
        self.miner_cash = miner_stats['miner_cash']
        self.current_view = MainMenu(self)  # Initialize current_view first
        root_container = self.current_view.box  # Use the box directly
        self.layout = Layout(root_container)
        self.pt_app = None  # Placeholder for the prompt_toolkit.Application instance

    def change_view(self, view):
        self.current_view = view
        self.layout.container = view.box  # Update the container directly
        self.layout.focus(view.box)  # Set focus on the new view's box

    def run(self):
        self.pt_app = prompt_toolkit.Application(
            layout=self.layout,
            key_bindings=bindings,
            full_screen=True
        )
        self.pt_app.custom_app = self  # Attach the custom Application instance
        self.pt_app.run()

if __name__ == "__main__":
    app = Application(predictions, gameData, miner_stats)
    app.run()
