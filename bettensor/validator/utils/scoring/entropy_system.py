import json
import traceback
import numpy as np
import math
import bittensor as bt
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
from scipy.spatial.distance import euclidean


class EntropySystem:
    state_file_path: str = "./bettensor/validator/state/entropy_system_state.json"
    def __init__(
        self,
        num_miners: int,
        max_days: int,
        
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
        self.load_state()

    def _get_array_for_day(self, array: np.ndarray, day: int) -> np.ndarray:
        return array[:, day % self.max_days]

    def _set_array_for_day(self, array: np.ndarray, day: int, values: np.ndarray):
        array[:, day % self.max_days] = values

    async def add_new_game(self, game_id: int, num_outcomes: int, odds: List[float]):
        if game_id in self.game_pools:
            bt.logging.warning(f"Game {game_id} already exists. Skipping.")
            return

        self.game_pools[game_id] = {}

        if len(odds) == 3 and odds[2] == 0.0:
            num_outcomes = 2
        else:
            num_outcomes = len(odds)

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

        # Mark the game as closed with timezone-aware datetime
        self.closed_games.add(game_id)
        self.game_close_times[game_id] = datetime.now(timezone.utc)
        bt.logging.info(f"Game {game_id} has been marked as closed.")

    def add_prediction(
        self, 
        prediction_id, 
        miner_uid, 
        game_id, 
        predicted_outcome, 
        wager, 
        predicted_odds, 
        prediction_date,
        historical_rebuild: bool = False
    ):
        """Add a prediction to the entropy system and calculate its contribution."""
        try:
            if game_id not in self.game_pools:
                bt.logging.error(f"Game {game_id} does not exist. Cannot add prediction {prediction_id}")
                return

            if predicted_outcome not in self.game_pools[game_id]:
                bt.logging.error(f"Invalid outcome {predicted_outcome} for game {game_id}")
                return

            # Only check for closed games during live operation, not during rebuilds
            if not historical_rebuild and game_id in self.closed_games:
                bt.logging.warning(f"Game {game_id} is closed. Cannot add prediction.")
                return

            # Ensure prediction_date is timezone-aware
            if isinstance(prediction_date, str):
                prediction_date = datetime.fromisoformat(prediction_date.replace('Z', '+00:00'))
            if prediction_date.tzinfo is None:
                prediction_date = prediction_date.replace(tzinfo=timezone.utc)

            entropy_contribution = self.calculate_entropy_contribution(
                game_id, predicted_outcome, miner_uid, predicted_odds, wager, prediction_date
            )
            
            if historical_rebuild:
                bt.logging.debug(
                    f"Adding historical prediction: game={game_id}, miner={miner_uid}, "
                    f"outcome={predicted_outcome}, contribution={entropy_contribution:.6f}"
                )
            else:
                bt.logging.debug(
                    f"Adding prediction: game={game_id}, miner={miner_uid}, "
                    f"outcome={predicted_outcome}, contribution={entropy_contribution:.6f}"
                )

            self.game_pools[game_id][predicted_outcome]["predictions"].append({
                "prediction_id": prediction_id,
                "miner_uid": miner_uid,
                "odds": predicted_odds,
                "wager": wager,
                "prediction_date": prediction_date,
                "entropy_contribution": entropy_contribution,
            })
            
        except Exception as e:
            bt.logging.error(f"Error in add_prediction: {e}")
            bt.logging.error(traceback.format_exc())

    def calculate_prediction_similarity(
        self, 
        game_id: int, 
        outcome: int, 
        miner_uid: int, 
        odds: float,
        prediction_date: datetime,
        wager: float
    ) -> float:
        """
        Calculate how similar this prediction is to existing ones.
        
        Args:
            game_id: The game identifier
            outcome: The predicted outcome
            miner_uid: The miner's identifier
            odds: The predicted odds
            prediction_date: The time of prediction
            wager: The amount wagered
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            pool = self.game_pools[game_id][outcome]
            existing_predictions = pool["predictions"]
            
            if not existing_predictions:
                return 0.0
                
            # Ensure prediction_date is timezone-aware
            if prediction_date.tzinfo is None:
                prediction_date = prediction_date.replace(tzinfo=timezone.utc)
                
            # Get all prediction times
            prediction_times = [
                pred["prediction_date"] if isinstance(pred["prediction_date"], datetime)
                else datetime.fromisoformat(pred["prediction_date"].replace('Z', '+00:00'))
                for pred in existing_predictions
            ]
            
            # Ensure all times are timezone-aware
            prediction_times = [
                pt.replace(tzinfo=timezone.utc) if pt.tzinfo is None else pt
                for pt in prediction_times
            ]
            
            # Add current prediction time
            all_times = prediction_times + [prediction_date]
            
            if len(all_times) < 2:
                return 0.0
                
            earliest_time = min(all_times)
            latest_time = max(all_times)
            
            # Calculate time range in seconds (add small buffer to avoid division by zero)
            time_range = max(
                (latest_time - earliest_time).total_seconds(),
                60  # minimum 1 minute range
            )
            
            # Time similarity (closer in time = more similar)
            time_similarity = 1 - abs(
                (prediction_date - earliest_time).total_seconds()
            ) / time_range
            
            # Odds similarity
            existing_odds = [float(pred["odds"]) for pred in existing_predictions]
            odds_range = max(max(existing_odds) - min(existing_odds), 0.1)  # minimum 0.1 range
            odds_similarity = 1 - abs(odds - np.mean(existing_odds)) / odds_range
            
            # Wager similarity
            existing_wagers = [float(pred["wager"]) for pred in existing_predictions]
            wager_range = max(max(existing_wagers) - min(existing_wagers), 1.0)  # minimum 1.0 range
            wager_similarity = 1 - abs(wager - np.mean(existing_wagers)) / wager_range
            
            # Combine similarities (weighted average)
            similarity = (
                0.4 * time_similarity +
                0.4 * odds_similarity +
                0.2 * wager_similarity
            )
            
            bt.logging.debug(
                f"Prediction similarity for game {game_id}: "
                f"time={time_similarity:.3f}, odds={odds_similarity:.3f}, "
                f"wager={wager_similarity:.3f}, combined={similarity:.3f}"
            )
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            bt.logging.error(f"Error calculating prediction similarity: {e}")
            bt.logging.error(traceback.format_exc())
            return 0.0

    def calculate_entropy_contribution(
        self, 
        game_id: int, 
        predicted_outcome: int, 
        miner_uid: int, 
        predicted_odds: float,
        wager: float,
        prediction_date: str
    ) -> float:
        """
        Calculate the entropy contribution for a prediction.
        
        Args:
            game_id: The game identifier
            predicted_outcome: The predicted outcome
            miner_uid: The miner's identifier
            predicted_odds: The predicted odds
            wager: The amount wagered
            prediction_date: The prediction datetime (ISO format string)
            
        Returns:
            float: The entropy contribution score
        """
        prediction_similarity = self.calculate_prediction_similarity(
            game_id=game_id,
            outcome=predicted_outcome,
            miner_uid=miner_uid,
            odds=predicted_odds,
            prediction_date=prediction_date,  # Fixed parameter order
            wager=wager
        )
        
        contrarian_component = self.calculate_contrarian_component(
            game_id, predicted_outcome, miner_uid
        )

        # Combine components with weights
        entropy_contribution = 0.6 * prediction_similarity + 0.4 * contrarian_component
        
        # Normalize to [-1, 1] range
        normalized_contribution = max(min(entropy_contribution, 1), -1)
        
        bt.logging.debug(
            f"Entropy contribution for game {game_id}: "
            f"similarity={prediction_similarity:.3f}, "
            f"contrarian={contrarian_component:.3f}, "
            f"final={normalized_contribution:.3f}"
        )
        
        return normalized_contribution

    def calculate_contrarian_component(self, game_id, predicted_outcome, miner_uid):
        total_predictions = sum(
            len(pool["predictions"]) for pool in self.game_pools[game_id].values()
        )
        if total_predictions == 0:
            return 0.5  # Neutral score for the first prediction

        outcome_predictions = len(self.game_pools[game_id][predicted_outcome]["predictions"])
        outcome_ratio = outcome_predictions / total_predictions

        # Direct inverse relationship
        contrarian_score = 1 - outcome_ratio

        # Ensure the score is between 0 and 1
        contrarian_score = max(0, min(1, contrarian_score))

        # Adjust the scale to make it more granular
        # If outcome_ratio is less than 0.1 (10%), contrarian_score will be >= 0.9
        adjusted_score = contrarian_score**0.5  # This makes the scale more granular

        bt.logging.debug(
            f"Contrarian calculation for game {game_id}, outcome {predicted_outcome}:"
        )
        bt.logging.debug(f"  Total predictions: {total_predictions}")
        bt.logging.debug(f"  Outcome predictions: {outcome_predictions}")
        bt.logging.debug(f"  Outcome ratio: {outcome_ratio:.4f}")
        bt.logging.debug(f"  Contrarian score: {adjusted_score:.4f}")

        return (
            adjusted_score - 0.5
        )  # Center around 0 for consistency with other components

    def calculate_final_entropy_score(self, miner_uid: int) -> float:
        """
        Calculate the final entropy score for a miner for the current day.

        Args:
            miner_id (int): ID of the miner.

        Returns:
            float: Final entropy score.
        """
        contributions = self.miner_predictions[self.current_day].get(miner_uid, [])
        final_score = sum(contributions)
        bt.logging.info(
            f"Final entropy score for miner {miner_uid} on day {self.current_day}: {final_score:.4f}"
        )
        return final_score

    def calculate_final_entropy_scores_for_miners(self) -> Dict[int, float]:
        """
        Calculate and return the final entropy scores for all miners for the current day.

        Returns:
            Dict[int, float]: Mapping from miner ID to their final entropy score.
        """
        final_scores = {}
        for miner_uid in range(self.num_miners):
            score = self.calculate_final_entropy_score(miner_uid)
            final_scores[miner_uid] = score
        return final_scores

    def reset_predictions_for_closed_games(self, current_date: datetime):
        """
        Reset predictions for games that are older than 45 days.
        
        Args:
            current_date (datetime): The current date to calculate age from.
        """
        # Ensure current_date is timezone-aware
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=timezone.utc)
            
        cutoff_date = current_date - timedelta(days=45)
        
        for game_id in list(self.closed_games):
            game_close_time = self.game_close_times.get(game_id)
            if game_close_time:
                # Ensure game_close_time is timezone-aware
                if game_close_time.tzinfo is None:
                    game_close_time = game_close_time.replace(tzinfo=timezone.utc)
                    
                if game_close_time < cutoff_date:
                    # Remove old predictions but keep the game structure
                    for outcome, pool in self.game_pools[game_id].items():
                        pool["predictions"].clear()
                        pool["entropy_score"] = 0.0
                        bt.logging.debug(f"Cleared old predictions for game {game_id}, outcome {outcome}")
                        
                    # Update tracking sets/dicts
                    self.closed_games.remove(game_id)
                    del self.game_close_times[game_id]
                    bt.logging.debug(f"Removed tracking for old game {game_id}")

    def calculate_initial_entropy(self, initial_odds: float) -> float:
        """
        Calculate the initial entropy for a game outcome based on the odds.

        Args:
            initial_odds (float): Odds for the outcome.

        Returns:
            float: Initial entropy value.
        """
        # Handle invalid odds (negative or zero)
        if initial_odds <= 0:
            return 0.0
        
        prob = 1 / (initial_odds + self.epsilon)
        # Ensure probability is in valid range [0,1]
        prob = max(0, min(1, prob))
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
        Get EBDR scores for all miners across the scoring window.
        """
        bt.logging.debug(f"Getting EBDR scores for day {current_day}")
        
        # Initialize scores array for all days
        ebdr_scores = np.zeros((self.num_miners, self.max_days))
        
        # First, calculate contributions for each miner from active games
        miner_contributions = defaultdict(float)
        
        for game_id in game_ids:
            if game_id in self.game_pools:
                for outcome, pool in self.game_pools[game_id].items():
                    for prediction in pool["predictions"]:
                        miner_uid = int(prediction["miner_uid"])
                        contribution = prediction["entropy_contribution"]
                        miner_contributions[miner_uid] += contribution
        
        # Store contributions in miner_predictions for the current day
        self.miner_predictions[current_day] = dict(miner_contributions)
        
        # Now populate the scores array using historical data
        for day in range(self.max_days):
            if day in self.miner_predictions:
                for miner_uid, score in self.miner_predictions[day].items():
                    ebdr_scores[miner_uid, day] = score
        
        # Normalize scores for each day
        for day in range(self.max_days):
            max_score = np.max(ebdr_scores[:, day])
            if max_score > 0:
                ebdr_scores[:, day] /= max_score
        
        bt.logging.info(f"EBDR scores calculated for day {current_day}")
        bt.logging.debug(f"Non-zero scores: {np.count_nonzero(ebdr_scores)}")
        bt.logging.debug(f"Max score: {np.max(ebdr_scores)}")
        bt.logging.debug(f"Mean score: {np.mean(ebdr_scores)}")
        
        # Clean up old predictions but keep last 45 days
        self.reset_predictions_for_closed_games(current_date)
        
        return ebdr_scores

    def save_state(self):
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
                                "miner_uid": pred["miner_uid"],
                                "odds": pred["odds"],
                                "wager": pred["wager"],
                                "prediction_date": pred["prediction_date"],
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

        with open(self.state_file_path, "w") as f:
            json.dump(state, f)

        bt.logging.info(f"EntropySystem state saved to {self.state_file_path}")

    def load_state(self):
        """
        Load the state of the EntropySystem from a JSON file.

        Args:
            file_path (str): The path to load the JSON file from.
        """
        try:
            with open(self.state_file_path, "r") as f:
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
                                "miner_uid": pred["miner_uid"],
                                "odds": pred["odds"],
                                "wager": pred["wager"],
                                "prediction_date": pred["prediction_date"],
                                "entropy_contribution": pred["entropy_contribution"],
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
            bt.logging.info(f"EntropySystem state loaded from {self.state_file_path}")

        except FileNotFoundError:
            bt.logging.warning(
                f"No state file found at {self.state_file_path}. Starting with fresh state."
            )

        except json.JSONDecodeError:
            bt.logging.error(
                f"Error decoding JSON from {self.state_file_path}. Starting with fresh state."
            )
            bt.logging.error(traceback.format_exc())
        except KeyError as e:
            bt.logging.error(
                f"Missing key in state file: {e}. Starting with fresh state."
            )
            bt.logging.error(traceback.format_exc())
        except Exception as e:
            bt.logging.error(
                f"Unexpected error loading state: {e}. Starting with fresh state."
            )
            bt.logging.error(traceback.format_exc())

    def reset_state(self):
        """Reset the entropy system state to initial values."""
        bt.logging.info("Resetting entropy system state")
        
        # Reset game pools
        self.game_pools = {}
        
        # Reset closed games tracking
        self.closed_games = set()
        
        # Reset game close times
        self.game_close_times = {}
        
        # Reset miner predictions tracking
        self.miner_predictions = {}
        
        # Reset any other state variables
        self.last_processed_date = None
        
        bt.logging.debug("Entropy system state has been reset")

    def get_state_summary(self):
        """Get a summary of the current entropy system state."""
        return {
            "num_active_games": len(self.game_pools) - len(self.closed_games),
            "num_closed_games": len(self.closed_games),
            "num_days_with_predictions": len(self.miner_predictions),
            "oldest_game_date": min(self.game_close_times.values()) if self.game_close_times else None,
            "newest_game_date": max(self.game_close_times.values()) if self.game_close_times else None
        }

    def log_state(self):
        """Log the current state of the entropy system."""
        state = self.get_state_summary()
        bt.logging.info("Entropy System State:")
        for key, value in state.items():
            bt.logging.info(f"  {key}: {value}")

