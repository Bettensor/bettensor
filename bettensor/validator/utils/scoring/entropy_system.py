import json
import torch as t
import math
import bittensor as bt
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

class EntropySystem:
    def __init__(self, max_capacity: int, max_days: int, state_file_path: str = 'entropy_system_state.json'):
        """
        Initialize the EntropySystem object.

        Args:
            max_capacity (int): Maximum number of miners.
            max_days (int): Maximum number of days to store scores.
            state_file_path (str, optional): Path to the state file. Defaults to 'entropy_system_state.json'.
        """
        self.max_capacity = max_capacity
        self.max_days = max_days
        self.ebdr_scores = t.zeros(max_capacity, max_days)
        self.current_day = 0
        self.game_outcome_entropies = {}
        self.prediction_counts = {}
        self.epsilon = 1e-8

        # Prediction Pools and Entropy Scores
        self.game_pools = defaultdict(dict)  # {game_id: {outcome: Pool}}
        self.pool_entropy_scores = defaultdict(dict)  # {game_id: {outcome: entropy_score}}
        self.closed_games = set()  # Set of game_ids that have been scored/closed

        # Miner Predictions Per Day
        self.miner_predictions = defaultdict(lambda: defaultdict(list))  # {day: {miner_id: [contributions]}}

        # Attempt to load existing state
        self.load_state(state_file_path)

    def add_new_game(self, game_id: int, num_outcomes: int):
        """
        Add a new game and initialize prediction pools based on the number of outcomes.

        Args:
            game_id (int): Unique identifier for the game.
            num_outcomes (int): Number of possible outcomes for the game.
        """
        if game_id in self.game_pools:
            bt.logging.warning(f"Game {game_id} already exists. Skipping creation of new pools.")
            return

        pools = {}
        entropy_scores = {}
        for outcome in range(num_outcomes):
            pool_id = f"pool_{outcome}"
            pools[outcome] = {
                "predictions": [],
                "entropy_score": self.calculate_initial_entropy(closing_line_odds=1.0)  # Initialize with default odds
            }
            entropy_scores[outcome] = pools[outcome]["entropy_score"]
            bt.logging.info(f"Initialized pool '{pool_id}' for game {game_id}, outcome {outcome} with entropy score {pools[outcome]['entropy_score']:.4f}")

        self.game_pools[game_id] = pools
        self.pool_entropy_scores[game_id] = entropy_scores
        bt.logging.info(f"Added new game {game_id} with {num_outcomes} outcomes.")

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

        # Clear prediction pools for the closed game
        for outcome, pool in self.game_pools[game_id].items():
            pool["predictions"].clear()
            self.pool_entropy_scores[game_id][outcome] = 0.0
            bt.logging.info(f"Cleared predictions for game {game_id}, outcome {outcome}.")

        # Mark the game as closed
        self.closed_games.add(game_id)
        bt.logging.info(f"Game {game_id} has been marked as closed.")

    def add_prediction(self, miner_id: int, game_id: int, outcome: int, odds: float, wager: float, prediction_time: datetime):
        """
        Add a prediction to the appropriate pool and update entropy scores.

        Args:
            miner_id (int): ID of the miner making the prediction.
            game_id (int): ID of the game.
            outcome (int): Predicted outcome.
            odds (float): Odds for the predicted outcome.
            wager (float): Amount wagered.
            prediction_time (datetime): Timestamp of the prediction.
        """
        if game_id in self.closed_games:
            bt.logging.warning(f"Game {game_id} is closed. Cannot add prediction.")
            return

        if game_id not in self.game_pools:
            bt.logging.error(f"Game {game_id} does not exist. Cannot add prediction.")
            return

        if outcome not in self.game_pools[game_id]:
            bt.logging.error(f"Outcome {outcome} does not exist for game {game_id}. Cannot add prediction.")
            return

        pool = self.game_pools[game_id][outcome]

        # Calculate entropy contribution
        entropy_contribution = self.calculate_entropy_contribution(game_id, outcome, miner_id, odds, wager, prediction_time)

        # Add prediction to the pool
        prediction = {
            "miner_id": miner_id,
            "odds": odds,
            "wager": wager,
            "prediction_time": prediction_time,
            "entropy_contribution": entropy_contribution
        }
        pool["predictions"].append(prediction)
        bt.logging.info(f"Added prediction by miner {miner_id} to game {game_id}, outcome {outcome} with entropy contribution {entropy_contribution:.4f}")

        # Update pool's entropy score
        self.pool_entropy_scores[game_id][outcome] += entropy_contribution
        bt.logging.info(f"Updated entropy score for game {game_id}, outcome {outcome}: {self.pool_entropy_scores[game_id][outcome]:.4f}")

        # Record miner's prediction for the current day
        self.miner_predictions[self.current_day][miner_id].append(entropy_contribution)

    def calculate_entropy_contribution(self, game_id: int, outcome: int, miner_id: int, odds: float, wager: float, prediction_time: datetime) -> float:
        """
        Calculate the entropy contribution of a prediction.

        Args:
            game_id (int): ID of the game.
            outcome (int): Predicted outcome.
            miner_id (int): ID of the miner.
            odds (float): Odds for the predicted outcome.
            wager (float): Amount wagered.
            prediction_time (datetime): Timestamp of the prediction.

        Returns:
            float: Entropy contribution score.
        """
        # Example components for entropy contribution
        # These can be adjusted based on specific requirements
        order_component = self.calculate_order_component(game_id, outcome, prediction_time)
        wager_similarity_component = self.calculate_wager_similarity(game_id, outcome, wager)
        contrarian_component = self.calculate_contrarian_component(game_id, outcome, miner_id)

        entropy_contribution = order_component + wager_similarity_component + contrarian_component
        return entropy_contribution

    def calculate_order_component(self, game_id: int, outcome: int, prediction_time: datetime) -> float:
        """
        Calculate the order/timing component of entropy.

        Args:
            game_id (int): ID of the game.
            outcome (int): Predicted outcome.
            prediction_time (datetime): Timestamp of the prediction.

        Returns:
            float: Order component score.
        """
        pool = self.game_pools[game_id][outcome]
        if not pool["predictions"]:
            return 0.5  # Neutral entropy for the first prediction

        # Assuming earlier predictions have higher entropy
        # Calculate based on the prediction_time compared to existing predictions
        earliest_time = min(pred["prediction_time"] for pred in pool["predictions"])
        time_diff = (prediction_time - earliest_time).total_seconds()
        normalized_time = max(1 - (time_diff / 3600), 0)  # Normalize to [0,1], assuming 1 hour window
        return normalized_time * 0.5  # Weighted accordingly

    def calculate_wager_similarity(self, game_id: int, outcome: int, wager: float) -> float:
        """
        Calculate the wager similarity component of entropy.

        Args:
            game_id (int): ID of the game.
            outcome (int): Predicted outcome.
            wager (float): Amount wagered.

        Returns:
            float: Wager similarity component score.
        """
        pool = self.game_pools[game_id][outcome]
        if not pool["predictions"]:
            return 0.5  # Neutral entropy for the first prediction

        average_wager = sum(pred["wager"] for pred in pool["predictions"]) / len(pool["predictions"])
        similarity = 1.0 - abs(wager - average_wager) / max(average_wager, self.epsilon)
        return similarity * 0.5  # Weighted accordingly

    def calculate_contrarian_component(self, game_id: int, outcome: int, miner_id: int) -> float:
        """
        Calculate the contrarian/uniqueness component of entropy.

        Args:
            game_id (int): ID of the game.
            outcome (int): Predicted outcome.
            miner_id (int): ID of the miner.

        Returns:
            float: Contrarian component score.
        """
        # Example: If miner's prediction is unique across all pools
        unique = True
        for out, pool in self.game_pools[game_id].items():
            if out == outcome:
                continue
            for pred in pool["predictions"]:
                if pred["miner_id"] == miner_id:
                    unique = False
                    break
            if not unique:
                break
        return 1.0 if unique else 0.0

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
        bt.logging.info(f"Final entropy score for miner {miner_id} on day {self.current_day}: {final_score:.4f}")
        return final_score

    def calculate_final_entropy_scores_for_miners(self) -> Dict[int, float]:
        """
        Calculate and return the final entropy scores for all miners for the current day.

        Returns:
            Dict[int, float]: Mapping from miner ID to their final entropy score.
        """
        final_scores = {}
        for miner_id in range(self.max_capacity):
            score = self.calculate_final_entropy_score(miner_id)
            final_scores[miner_id] = score
        return final_scores

    def reset_predictions_for_closed_games(self):
        """
        Reset the miner_predictions for games that have been closed.
        """
        for game_id in list(self.closed_games):
            for outcome, pool in self.game_pools[game_id].items():
                pool["predictions"].clear()
                self.pool_entropy_scores[game_id][outcome] = 0.0
                bt.logging.info(f"Cleared predictions for closed game {game_id}, outcome {outcome}.")
            self.closed_games.remove(game_id)
            bt.logging.info(f"Reset predictions for closed game {game_id}.")

    def update_ebdr_scores(self, predictions, closing_line_odds, results):
        """
        Update the EBDR scores based on predictions, closing line odds, and results.

        Args:
            predictions (list): List of miner predictions.
            closing_line_odds (torch.Tensor): Tensor of closing line odds.
            results (torch.Tensor): Tensor of game results.

        Returns:
            torch.Tensor: Updated EBDR scores.
        """
        if not predictions:
            bt.logging.warning("No predictions provided. Skipping EBDR score update.")
            return self.ebdr_scores

        entropy_scores = self.ebdr_scores[:, self.current_day].clone()
        bt.logging.info(f"Initial entropy_scores: {entropy_scores}")

        for game_id in range(closing_line_odds.shape[0]):
            if game_id not in self.game_outcome_entropies:
                self.game_outcome_entropies[game_id] = {}
                self.prediction_counts[game_id] = {}
                outcomes = closing_line_odds[game_id][1:].size(0)  # Number of outcomes
                for outcome in range(outcomes):
                    initial_entropy = self.calculate_initial_entropy(closing_line_odds[game_id][1:][outcome])
                    self.game_outcome_entropies[game_id][outcome] = initial_entropy
                    self.prediction_counts[game_id][outcome] = 0
                    bt.logging.info(f"Initialized entropy for game {game_id}, outcome {outcome}: {initial_entropy}")

        for pred in predictions:
            miner_id, game_id, outcome, odds, wager, prediction_time = pred
            miner_id = int(miner_id.item())
            game_id = int(game_id.item())
            outcome = int(outcome.item())
            odds = float(odds.item())
            wager = float(wager.item())
            prediction_time = prediction_time.to_pydatetime()

            self.add_prediction(miner_id, game_id, outcome, odds, wager, prediction_time)

        # Update the ebdr_scores with the new entropy scores
        self.ebdr_scores[:, self.current_day] = entropy_scores
        bt.logging.info(f"Updated self.ebdr_scores for day {self.current_day}")
        bt.logging.info(f"Entropy scores: {self.ebdr_scores}")

        return self.ebdr_scores

    def calculate_initial_entropy(self, closing_line_odds: float) -> float:
        """
        Calculate the initial entropy for a game outcome based on the odds.

        Args:
            closing_line_odds (float): Odds for the outcome.

        Returns:
            float: Initial entropy value.
        """
        prob = 1 / (closing_line_odds + self.epsilon)
        entropy = -prob * math.log2(prob + self.epsilon)
        result = max(entropy, self.epsilon)
        bt.logging.debug(f"Initial entropy calculation: odds={closing_line_odds}, result={result}")
        return result

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
            "pool_entropy_scores": {
                str(game_id): {str(outcome): score for outcome, score in outcomes.items()}
                for game_id, outcomes in self.pool_entropy_scores.items()
            },
            "miner_predictions": {
                str(day): {str(miner_id): contributions for miner_id, contributions in miners.items()}
                for day, miners in self.miner_predictions.items()
            },
            "game_pools": {
                str(game_id): {
                    str(outcome): {
                        "predictions": [
                            {
                                "miner_id": pred["miner_id"],
                                "odds": pred["odds"],
                                "wager": pred["wager"],
                                "prediction_time": pred["prediction_time"].isoformat(),
                                "entropy_contribution": pred["entropy_contribution"]
                            } for pred in pool["predictions"]
                        ],
                        "entropy_score": pool["entropy_score"]
                    } for outcome, pool in outcomes.items()
                } for game_id, outcomes in self.game_pools.items()
            },
            "closed_games": list(self.closed_games)
        }

        with open(file_path, 'w') as f:
            json.dump(state, f)

        bt.logging.info(f"EntropySystem state saved to {file_path}")

    def load_state(self, file_path):
        """
        Load the state of the EntropySystem from a JSON file.

        Args:
            file_path (str): The path to load the JSON file from.
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)

            self.current_day = int(state["current_day"])
            self.game_outcome_entropies = {int(k): {int(ok): v for ok, v in ov.items()} for k, ov in state["game_outcome_entropies"].items()}
            self.prediction_counts = {int(k): {int(ok): v for ok, v in ov.items()} for k, ov in state["prediction_counts"].items()}
            self.ebdr_scores = t.tensor(state["ebdr_scores"])
            self.pool_entropy_scores = {
                int(game_id): {int(outcome): float(score) for outcome, score in outcomes.items()}
                for game_id, outcomes in state["pool_entropy_scores"].items()
            }
            self.miner_predictions = {
                int(day): {int(miner_id): float(contribution) for miner_id, contribution in miners.items()}
                for day, miners in state["miner_predictions"].items()
            }
            self.game_pools = defaultdict(dict)
            for game_id, outcomes in state["game_pools"].items():
                for outcome, pool in outcomes.items():
                    self.game_pools[int(game_id)][int(outcome)] = {
                        "predictions": [
                            {
                                "miner_id": pred["miner_id"],
                                "odds": pred["odds"],
                                "wager": pred["wager"],
                                "prediction_time": datetime.fromisoformat(pred["prediction_time"]),
                                "entropy_contribution": float(pred["entropy_contribution"])
                            } for pred in pool["predictions"]
                        ],
                        "entropy_score": float(pool["entropy_score"])
                    }

            self.closed_games = set(state.get("closed_games", []))
            bt.logging.info(f"EntropySystem state loaded from {file_path}")

        except FileNotFoundError:
            bt.logging.warning(f"No state file found at {file_path}. Starting with fresh state.")
        except json.JSONDecodeError:
            bt.logging.error(f"Error decoding JSON from {file_path}. Starting with fresh state.")
        except KeyError as e:
            bt.logging.error(f"Missing key in state file: {e}. Starting with fresh state.")
        except Exception as e:
            bt.logging.error(f"Unexpected error loading state: {e}. Starting with fresh state.")

    def scoring_run(self, date, invalid_uids, valid_uids):
        """
        Example scoring run method integrating entropy scores.
        This method should be integrated with the main scoring system.

        Args:
            date (datetime): The date for which to run the scoring.
            invalid_uids (List[int]): List of invalid user IDs.
            valid_uids (List[int]): List of valid user IDs.

        Returns:
            torch.Tensor: Calculated weights.
        """
        bt.logging.info(f"=== Starting entropy scoring run for date: {date.isoformat()} ===")

        # Example logic to close games that have ended by 'date'
        # This should be replaced with actual game status checks
        closed_game_ids = self.identify_closed_games(date)
        for game_id in closed_game_ids:
            self.close_game(game_id)

        # Reset predictions only for closed games
        self.reset_predictions_for_closed_games()

        # Calculate final entropy scores for all miners
        final_entropy_scores = self.calculate_final_entropy_scores_for_miners()

        # Update ebdr_scores tensor
        for miner_id, score in final_entropy_scores.items():
            if miner_id < self.max_capacity:
                self.ebdr_scores[miner_id, self.current_day] = score

        bt.logging.info(f"=== Completed entropy scoring run for date: {date.isoformat()} ===")

        # Save state at the end of the run
        self.save_state('entropy_system_state.json')

        # Optionally, return the entropy scores
        return self.ebdr_scores[:, self.current_day]

    def identify_closed_games(self, current_date: datetime) -> List[int]:
        """
        Identify which games should be closed based on the current date.
        This is a placeholder method and should be implemented based on actual game data.

        Args:
            current_date (datetime): The current date.

        Returns:
            List[int]: List of game IDs to be closed.
        """
        # Placeholder implementation:
        # Return a list of game_ids whose end date is <= current_date
        # This requires access to game schedules, which is not defined here
        # For demonstration, assume no games are closed
        return []
