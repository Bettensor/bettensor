import torch as t
import math
import bittensor as bt

class EntropySystem:
    def __init__(self, max_capacity: int, max_days: int):
        """
        Initialize the EntropySystem object.

        Args:
            max_capacity (int): Maximum number of miners.
            max_days (int): Maximum number of days to store scores.
        """
        self.max_capacity = max_capacity
        self.max_days = max_days
        self.ebdr_scores = t.zeros(max_capacity, max_days)
        self.current_day = 0
        self.game_outcome_entropies = {}  # Store entropy for each outcome of each game
        self.prediction_counts = {}       # Store prediction counts per outcome of each game
        self.epsilon = 1e-8              # Small constant to avoid log(0)

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

        for i, miner_predictions in enumerate(predictions):
            if miner_predictions.numel() > 0:
                miner_entropy = 0
                valid_predictions = 0
                for pred in miner_predictions:
                    game_id, outcome, odds, wager = pred
                    game_id = int(game_id.item())
                    outcome = int(outcome.item())
                    if game_id < closing_line_odds.shape[0]:
                        new_entropy = self.update_game_entropy(game_id, outcome, odds)
                        miner_entropy += new_entropy
                        valid_predictions += 1
                        self.prediction_counts[game_id][outcome] += 1

                if valid_predictions > 0:
                    entropy_scores[i] = miner_entropy / valid_predictions
                else:
                    entropy_scores[i] = self.epsilon
                    bt.logging.info(f"Miner {i}: No valid predictions, setting entropy to epsilon {self.epsilon}")
            else:
                bt.logging.info(f"Miner {i} has no predictions")


        self.ebdr_scores[:, self.current_day] = entropy_scores
        bt.logging.info(f"Updated self.ebdr_scores for day {self.current_day}")
        bt.logging.info(f"Entropy scores: {self.ebdr_scores}")

        return self.ebdr_scores

    def calculate_initial_entropy(self, odds):
        """
        Calculate the initial entropy for a game outcome based on the odds.

        Args:
            odds (float): Odds for the outcome.

        Returns:
            float: Initial entropy value.
        """
        prob = 1 / (odds + self.epsilon)
        entropy = -prob * math.log2(prob + self.epsilon)
        result = max(entropy, self.epsilon)
        bt.logging.debug(f"Initial entropy calculation: odds={odds}, result={result}")
        return result

    def update_game_entropy(self, game_id, outcome, odds):
        """
        Update the entropy for a specific game outcome based on new predictions.

        Args:
            game_id (int): ID of the game.
            outcome (int): Outcome index.
            odds (float): Odds for the outcome.

        Returns:
            float: Updated entropy value.
        """
        bt.logging.trace(f"EntropySystem.update_game_entropy()| Updating game {game_id} outcome {outcome} entropy")
        current_entropy = self.game_outcome_entropies[game_id].get(outcome, self.epsilon)
        prob = 1 / (odds + self.epsilon)
        new_entropy = -prob * math.log2(prob + self.epsilon)
        updated_entropy = max((current_entropy + new_entropy) / 2, self.epsilon)
        bt.logging.trace(f"EntropySystem.update_game_entropy()| Updated game {game_id} outcome {outcome} entropy: {updated_entropy}")
        self.game_outcome_entropies[game_id][outcome] = updated_entropy
        bt.logging.debug(
            f"Updating game {game_id} outcome {outcome} entropy: current={current_entropy}, new={new_entropy}, updated={updated_entropy}"
        )
        return updated_entropy

    def calculate_bookmaker_probabilities(self, odds):
        """
        Calculate the bookmaker probabilities based on the odds.

        Args:
            odds (torch.Tensor): Tensor of odds.

        Returns:
            torch.Tensor: Tensor of calculated probabilities.
        """
        probs = 1 / odds
        return probs / probs.sum()

    def calculate_event_entropy(self, event):
        """
        Calculate the entropy for a specific event.

        Args:
            event (dict): Dictionary containing event data.

        Returns:
            tuple: Miner entropy and bookmaker entropy.
        """
        miner_probs = []
        for pred in event["predictions"].values():
            if pred["odds"] > 0:
                miner_probs.append(1 / pred["odds"])
            else:
                bt.logging.warning(
                    f"Invalid miner odd ({pred['odds']}) encountered. Using default probability."
                )
                miner_probs.append(0.01)  # Use a small default probability
        miner_probs = t.tensor(miner_probs)
        miner_probs = miner_probs / miner_probs.sum()

        bookmaker_probs = self.calculate_bookmaker_probabilities(event["current_odds"])

        miner_entropy = -t.sum(miner_probs * t.log2(miner_probs + 1e-10))
        bookmaker_entropy = -t.sum(bookmaker_probs * t.log2(bookmaker_probs + 1e-10))

        return miner_entropy.item(), bookmaker_entropy.item()

    def calculate_miner_entropy_contribution(self, miner_id, event):
        """
        Calculate the entropy contribution of a specific miner for an event.

        Args:
            miner_id (int): ID of the miner.
            event (dict): Dictionary containing event data.

        Returns:
            float: Entropy contribution of the miner.
        """
        miner_prob = 1 / event["predictions"][miner_id]["odds"]
        return -miner_prob * math.log2(miner_prob + 1e-10)

    def calculate_consensus_prediction(self, event):
        """
        Calculate the consensus prediction for an event.

        Args:
            event (dict): Dictionary containing event data.

        Returns:
            dict: Consensus prediction containing outcome and odds.
        """
        consensus = {"outcome": 0, "odds": 0}
        total_wager = sum(pred["wager_size"] for pred in event["predictions"].values())

        for pred in event["predictions"].values():
            weight = pred["wager_size"] / total_wager
            consensus["outcome"] += weight * pred["outcome"]
            consensus["odds"] += weight * pred["odds"]

        return consensus

    def calculate_uniqueness_score(self, prediction, consensus):
        """
        Calculate the uniqueness score of a prediction compared to the consensus.

        Args:
            prediction (dict): Dictionary containing prediction data.
            consensus (dict): Dictionary containing consensus prediction data.

        Returns:
            float: Uniqueness score.
        """
        outcome_diff = abs(prediction["outcome"] - consensus["outcome"])
        odds_diff = abs(prediction["odds"] - consensus["odds"]) / max(
            prediction["odds"], consensus["odds"]
        )
        return (outcome_diff + odds_diff) / 2

    def update_prediction_history(self, miner_id, prediction):
        """
        Update the prediction history for a specific miner.

        Args:
            miner_id (int): ID of the miner.
            prediction (dict): Dictionary containing prediction data.
        """
        if miner_id not in self.prediction_history:
            self.prediction_history[miner_id] = []
        self.prediction_history[miner_id].append(prediction)
        if len(self.prediction_history[miner_id]) > self.entropy_window:
            self.prediction_history[miner_id].pop(0)

    def calculate_historical_uniqueness(self, miner_id):
        """
        Calculate the historical uniqueness score for a specific miner.

        Args:
            miner_id (int): ID of the miner.

        Returns:
            float: Historical uniqueness score.
        """
        if (
            miner_id not in self.prediction_history
            or not self.prediction_history[miner_id]
        ):
            return 0.0

        miner_history = self.prediction_history[miner_id]
        all_histories = list(self.prediction_history.values())
        if not all_histories:
            return 0.0

        uniqueness = sum(1 for h in all_histories if h != miner_history) / len(
            all_histories
        )
        return uniqueness

    def calculate_contrarian_bonus(
        self, miner_prediction, consensus_prediction, actual_outcome
    ):
        """
        Calculate the contrarian bonus for a miner's prediction.

        Args:
            miner_prediction (dict): Dictionary containing miner's prediction data.
            consensus_prediction (dict): Dictionary containing consensus prediction data.
            actual_outcome (float): Actual outcome of the event.

        Returns:
            float: Contrarian bonus.
        """
        if (
            miner_prediction["outcome"] != consensus_prediction["outcome"]
            and miner_prediction["outcome"] == actual_outcome
        ):
            return 1.5
        return 1.0

    def get_ebdr_scores(self):
        """
        Get the EBDR scores.

        Returns:
            torch.Tensor: EBDR scores.
        """
        return self.ebdr_scores

    def get_current_ebdr_scores(self):
        """
        Get the current EBDR scores for the current day.

        Returns:
            torch.Tensor: Current EBDR scores.
        """
        return self.ebdr_scores[:, self.current_day]

    def get_uniqueness_scores(self):
        """
        Get the uniqueness scores for all miners.

        Returns:
            torch.Tensor: Uniqueness scores.
        """
        uniqueness_scores = t.zeros(self.max_capacity)
        for miner_id in range(self.max_capacity):
            uniqueness_scores[miner_id] = self.calculate_historical_uniqueness(
                str(miner_id)
            )
        return uniqueness_scores

    def get_contrarian_bonuses(self):
        """
        Get the contrarian bonuses for all miners.

        Returns:
            torch.Tensor: Contrarian bonuses.
        """
        return t.ones(self.max_capacity)

    def get_historical_uniqueness(self):
        """
        Get the historical uniqueness scores for all miners.

        Returns:
            torch.Tensor: Historical uniqueness scores.
        """
        return self.get_uniqueness_scores()
