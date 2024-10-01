import json
import numpy as np
import math
import bittensor as bt
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
from scipy.spatial.distance import euclidean


class EntropySystem:
    def __init__(
        self,
        num_miners: int,
        max_days: int,
        state_file_path: str = "entropy_system_state.json",
    ):
        """
        Initialize the EntropySystem object.

        Args:
            num_miners (int): Maximum number of miners.
            max_days (int): Maximum number of days to store scores.
            state_file_path (str, optional): Path to the state file. Defaults to 'entropy_system_state.json'.
        """
        self.num_miners = num_miners
        self.max_days = max_days
        self.current_day = 0
        self.ebdr_scores = np.zeros((num_miners, max_days))
        self.game_outcome_entropies = {}
        self.prediction_counts = {}
        self.game_pools = defaultdict(
            lambda: defaultdict(lambda: {"predictions": [], "entropy_score": 0.0})
        )
        self.miner_predictions = defaultdict(lambda: defaultdict(list))
        self.epsilon = 1e-8
        self.closed_games = set()
        self.game_close_times = {}

        # Attempt to load existing state
        self.load_state(state_file_path)

    def _get_array_for_day(self, array: np.ndarray, day: int) -> np.ndarray:
        return array[:, day % self.max_days]

    def _set_array_for_day(self, array: np.ndarray, day: int, values: np.ndarray):
        array[:, day % self.max_days] = values

    def add_new_game(self, game_id: int, num_outcomes: int, odds: List[float]):
        if game_id in self.game_pools:
            bt.logging.warning(f"Game {game_id} already exists. Skipping.")
            return

        self.game_pools[game_id] = {}
        for i in range(num_outcomes):
            if i < len(odds):
                self.game_pools[game_id][i] = {
                    "predictions": [],
                    "entropy_score": self.calculate_initial_entropy(odds[i]),
                }
            else:
                bt.logging.warning(
                    f"Odds not provided for outcome {i} in game {game_id}. Outcome pool not initialized."
                )
                # Outcome pool is not initialized

        bt.logging.debug(
            f"Added new game {game_id} with {num_outcomes} outcomes. Odds: {odds}"
        )

    def close_game(self, game_id: int):
        """
        Mark a game as closed and clear its prediction pools.

        Args:
            game_id (int): Unique identifier for the game.
        """
        if game_id not in self.game_pools:
            bt.logging.error(f"Game {game_id} does not exist. Cannot close.")
            return

        if game_id in self.closed_games:
            bt.logging.warning(f"Game {game_id} is already closed.")
            return

        # Mark the game as closed and record the closing time
        self.closed_games.add(game_id)
        self.game_close_times[game_id] = datetime.now(timezone.utc)
        bt.logging.info(f"Game {game_id} has been marked as closed.")

    def add_prediction(self, miner_id, game_id, outcome, odds, wager, prediction_time):
        if game_id not in self.game_pools:
            bt.logging.error(f"Game {game_id} does not exist. Cannot add prediction.")
            return

        if outcome not in self.game_pools[game_id]:
            bt.logging.error(
                f"Invalid outcome {outcome} for game {game_id}. Available outcomes: {list(self.game_pools[game_id].keys())}. Cannot add prediction."
            )
            return

        if game_id in self.closed_games:
            bt.logging.warning(f"Game {game_id} is closed. Cannot add prediction.")
            return

        # Convert prediction_time to datetime object if it's a string
        if isinstance(prediction_time, str):
            prediction_time = datetime.fromisoformat(prediction_time).replace(
                tzinfo=timezone.utc
            )

        entropy_contribution = self.calculate_entropy_contribution(
            game_id, outcome, miner_id, odds, wager, prediction_time
        )

        self.game_pools[game_id][outcome]["predictions"].append(
            {
                "miner_id": miner_id,
                "odds": odds,
                "wager": wager,
                "prediction_time": prediction_time,
                "entropy_contribution": entropy_contribution,
            }
        )

        bt.logging.debug(
            f"Added prediction for game {game_id}, outcome {outcome} by miner {miner_id}"
        )

    def calculate_prediction_similarity(
        self, game_id, outcome, miner_id, wager, prediction_time
    ):
        predictions = self.game_pools[game_id][outcome]["predictions"]
        if not predictions:
            return 0.0

        # Ensure prediction_time is a datetime object
        if isinstance(prediction_time, str):
            prediction_time = datetime.fromisoformat(prediction_time).replace(
                tzinfo=timezone.utc
            )

        # Calculate time-based similarity
        prediction_times = [
            p["prediction_time"] for p in predictions if p["miner_id"] != miner_id
        ]
        if prediction_times:
            earliest_time = min(prediction_times)
            latest_time = max(prediction_times)
            time_range = (latest_time - earliest_time).total_seconds() + self.epsilon
            time_similarity = (
                1 - abs((prediction_time - earliest_time).total_seconds()) / time_range
            )
        else:
            time_similarity = 1.0

        # Calculate wager-based similarity
        wagers = [p["wager"] for p in predictions if p["miner_id"] != miner_id]
        if wagers:
            min_wager = min(wagers)
            max_wager = max(wagers)
            wager_range = max_wager - min_wager + self.epsilon
            wager_similarity = 1 - abs(wager - min_wager) / wager_range
        else:
            wager_similarity = 1.0

        # Combine similarities
        return (time_similarity + wager_similarity) / 2

    def calculate_entropy_contribution(
        self, game_id, outcome, miner_id, odds, wager, prediction_time
    ):
        prediction_similarity = self.calculate_prediction_similarity(
            game_id, outcome, miner_id, wager, prediction_time
        )
        contrarian_component = self.calculate_contrarian_component(
            game_id, outcome, miner_id
        )

        # Combine components with weights
        entropy_contribution = 0.6 * prediction_similarity + 0.4 * contrarian_component

        # Normalize to a reasonable range, e.g., [-1, 1]
        normalized_contribution = max(min(entropy_contribution, 1), -1)

        # bt.logging.trace(f"Entropy components for game {game_id}, outcome {outcome}, miner {miner_id}:")
        # bt.logging.trace(f"  Prediction Similarity: {prediction_similarity:.4f}")
        # bt.logging.trace(f"  Contrarian: {contrarian_component:.4f}")
        # bt.logging.trace(f"  Final Contribution: {normalized_contribution:.4f}")

        return normalized_contribution

    def calculate_contrarian_component(self, game_id, outcome, miner_id):
        total_predictions = sum(
            len(pool["predictions"]) for pool in self.game_pools[game_id].values()
        )
        if total_predictions == 0:
            return 0.5  # Neutral score for the first prediction

        outcome_predictions = len(self.game_pools[game_id][outcome]["predictions"])
        outcome_ratio = outcome_predictions / total_predictions

        # Direct inverse relationship
        contrarian_score = 1 - outcome_ratio

        # Ensure the score is between 0 and 1
        contrarian_score = max(0, min(1, contrarian_score))

        # Adjust the scale to make it more granular
        # If outcome_ratio is less than 0.1 (10%), contrarian_score will be >= 0.9
        adjusted_score = contrarian_score**0.5  # This makes the scale more granular

        bt.logging.debug(
            f"Contrarian calculation for game {game_id}, outcome {outcome}:"
        )
        bt.logging.debug(f"  Total predictions: {total_predictions}")
        bt.logging.debug(f"  Outcome predictions: {outcome_predictions}")
        bt.logging.debug(f"  Outcome ratio: {outcome_ratio:.4f}")
        bt.logging.debug(f"  Contrarian score: {adjusted_score:.4f}")

        return (
            adjusted_score - 0.5
        )  # Center around 0 for consistency with other components

    def calculate_final_entropy_score(self, miner_id: int) -> float:
        """
        Calculate the final entropy score for a miner for the current day.

        Args:
            miner_id (int): ID of the miner.

        Returns:
            float: Final entropy score.
        """
        contributions = self.miner_predictions[self.current_day].get(miner_id, [])
        final_score = sum(contributions)
        bt.logging.info(
            f"Final entropy score for miner {miner_id} on day {self.current_day}: {final_score:.4f}"
        )
        return final_score

    def calculate_final_entropy_scores_for_miners(self) -> Dict[int, float]:
        """
        Calculate and return the final entropy scores for all miners for the current day.

        Returns:
            Dict[int, float]: Mapping from miner ID to their final entropy score.
        """
        final_scores = {}
        for miner_id in range(self.num_miners):
            score = self.calculate_final_entropy_score(miner_id)
            final_scores[miner_id] = score
        return final_scores

    def reset_predictions_for_closed_games(self):
        """
        Reset the miner_predictions for games that have been closed more than 1 day ago.
        """
        current_time = datetime.now(timezone.utc)
        for game_id in list(self.closed_games):
            game_close_time = self.game_close_times.get(game_id)
            if game_close_time and (current_time - game_close_time) > timedelta(days=1):
                for outcome, pool in self.game_pools[game_id].items():
                    pool["predictions"].clear()
                    pool["entropy_score"] = 0.0
                    bt.logging.info(
                        f"Cleared predictions for closed game {game_id}, outcome {outcome}."
                    )
                self.closed_games.remove(game_id)
                del self.game_close_times[game_id]
                bt.logging.info(f"Reset predictions for closed game {game_id}.")

    def calculate_initial_entropy(self, initial_odds: float) -> float:
        """
        Calculate the initial entropy for a game outcome based on the odds.

        Args:
            closing_line_odds (float): Odds for the outcome.

        Returns:
            float: Initial entropy value.
        """
        prob = 1 / (initial_odds + self.epsilon)
        entropy = -prob * math.log2(prob + self.epsilon)
        result = max(entropy, self.epsilon)
        bt.logging.debug(
            f"Initial entropy calculation: odds={initial_odds}, result={result}"
        )
        return result

    def get_current_ebdr_scores(
        self, current_date: datetime, current_day: int, game_ids: List[int]
    ) -> np.ndarray:
        """
        Get the current day's EBDR scores for all miners.

        Args:
            current_date (datetime): The current date.
            current_day (int): The current day index.
            game_ids (List[int]): List of game IDs to consider.

        Returns:
            np.ndarray: Current day's EBDR scores for all miners.
        """

        bt.logging.debug(f"Getting current EBDR scores for day {current_day}")

        ebdr_scores = np.zeros(self.num_miners)

        for game_id in game_ids:
            # bt.logging.debug(f"Game {game_id}")
            if game_id in self.game_pools:
                # bt.logging.debug(f"Game {game_id} exists in game_pools")
                for outcome, pool in self.game_pools[game_id].items():
                    for prediction in pool["predictions"]:
                        # bt.logging.debug(f"Adding entropy contribution for game {game_id}, outcome {outcome}, miner {prediction['miner_id']}")
                        miner_id = prediction["miner_id"]
                        ebdr_scores[miner_id] += prediction["entropy_contribution"]

        # Normalize scores
        max_score = np.max(ebdr_scores)
        if max_score > 0:
            ebdr_scores /= max_score

        self.reset_predictions_for_closed_games()
        # bt.logging.info(
        #     f"EBDR scores - min: {ebdr_scores.min():.4f}, "
        #     f"max: {ebdr_scores.max():.4f}, "
        #     f"mean: {ebdr_scores.mean():.4f}, "
        #     f"non-zero: {np.count_nonzero(ebdr_scores)}"
        # )

        return ebdr_scores

    def save_state(self, file_path):
        """
        Save the current state of the EntropySystem to a JSON file.

        Args:
            file_path (str): The path to save the JSON file.
        """
        state = {
            "current_day": self.current_day,
            "game_outcome_entropies": self.game_outcome_entropies,
            "prediction_counts": self.prediction_counts,
            "ebdr_scores": self.ebdr_scores.tolist(),
            "game_pools": {
                str(game_id): {
                    str(outcome): {
                        "predictions": [
                            {
                                "miner_id": pred["miner_id"],
                                "odds": pred["odds"],
                                "wager": pred["wager"],
                                "prediction_time": pred["prediction_time"].isoformat(),
                                "entropy_contribution": pred["entropy_contribution"],
                            }
                            for pred in pool["predictions"]
                        ],
                        "entropy_score": pool["entropy_score"],
                    }
                    for outcome, pool in outcomes.items()
                }
                for game_id, outcomes in self.game_pools.items()
            },
            "miner_predictions": {
                str(day): {
                    str(miner_id): contributions
                    for miner_id, contributions in miners.items()
                }
                for day, miners in self.miner_predictions.items()
            },
            "closed_games": list(self.closed_games),
            "game_close_times": {
                str(game_id): game_close_time.isoformat()
                for game_id, game_close_time in self.game_close_times.items()
            },
        }

        with open(file_path, "w") as f:
            json.dump(state, f)

        bt.logging.info(f"EntropySystem state saved to {file_path}")

    def load_state(self, file_path):
        """
        Load the state of the EntropySystem from a JSON file.

        Args:
            file_path (str): The path to load the JSON file from.
        """
        try:
            with open(file_path, "r") as f:
                state = json.load(f)

            self.current_day = int(state["current_day"])
            self.game_outcome_entropies = {
                int(k): {int(ok): v for ok, v in ov.items()}
                for k, ov in state["game_outcome_entropies"].items()
            }
            self.prediction_counts = {
                int(k): {int(ok): v for ok, v in ov.items()}
                for k, ov in state["prediction_counts"].items()
            }
            self.ebdr_scores = np.array(state["ebdr_scores"])
            self.game_pools = defaultdict(dict)
            for game_id, outcomes in state["game_pools"].items():
                for outcome, pool in outcomes.items():
                    self.game_pools[int(game_id)][int(outcome)] = {
                        "predictions": [
                            {
                                "miner_id": pred["miner_id"],
                                "odds": pred["odds"],
                                "wager": pred["wager"],
                                "prediction_time": datetime.fromisoformat(
                                    pred["prediction_time"]
                                ),
                                "entropy_contribution": float(
                                    pred["entropy_contribution"]
                                ),
                            }
                            for pred in pool["predictions"]
                        ],
                        "entropy_score": float(pool["entropy_score"]),
                    }
            self.miner_predictions = {
                int(day): {
                    int(miner_id): float(contribution)
                    for miner_id, contribution in miners.items()
                }
                for day, miners in state["miner_predictions"].items()
            }
            self.closed_games = set(state.get("closed_games", []))
            self.game_close_times = {
                int(game_id): datetime.fromisoformat(game_close_time)
                for game_id, game_close_time in state.get(
                    "game_close_times", {}
                ).items()
            }
            bt.logging.info(f"EntropySystem state loaded from {file_path}")

        except FileNotFoundError:
            bt.logging.warning(
                f"No state file found at {file_path}. Starting with fresh state."
            )
        except json.JSONDecodeError:
            bt.logging.error(
                f"Error decoding JSON from {file_path}. Starting with fresh state."
            )
        except KeyError as e:
            bt.logging.error(
                f"Missing key in state file: {e}. Starting with fresh state."
            )
        except Exception as e:
            bt.logging.error(
                f"Unexpected error loading state: {e}. Starting with fresh state."
            )
