from dataclasses import dataclass
import redis
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys
import os


@dataclass
class MinerConfig:
    model_prediction: bool = False


def create_table(config):
    table = Table(title="Miner Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Model Prediction", "ON" if config.model_prediction else "OFF")
    return table


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def run_interface(config, redis_client):
    console = Console()

    while True:
        clear_screen()
        table = create_table(config)
        panel = Panel(table, title="Press 'm' to toggle Model Prediction, 'q' to quit")
        console.print(panel)

        key = console.input(">").lower()
        if key == "m":
            config.model_prediction = not config.model_prediction
            redis_client.set("model_prediction", str(config.model_prediction))
        elif key == "q":
            break

    console.print("Exiting...")


def main():
    config = MinerConfig()
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    run_interface(config, redis_client)


if __name__ == "__main__":
    main()
