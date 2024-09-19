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
        self.scores_per_day = 24  # Assuming hourly updates
        self.ebdr_scores = t.zeros(max_capacity, max_days, self.scores_per_day)
        self.current_day = 0
        self.current_hour = 0
        self.game_entropies = {}  # Store entropy for each game
        self.epsilon = 1e-8  # Small constant to avoid log(0)

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

        entropy_scores = self.ebdr_scores[
            :, self.current_day, self.current_hour
        ].clone()

        bt.logging.debug(f"Number of predictions: {len(predictions)}")
        bt.logging.debug(f"Closing line odds shape: {closing_line_odds.shape}")
        bt.logging.debug(f"Results shape: {results.shape}")

        # Initialize game entropies if not already present
        for game_id in range(closing_line_odds.shape[0]):
            if game_id not in self.game_entropies:
                initial_entropy = self.calculate_initial_entropy(
                    closing_line_odds[game_id]
                )
                self.game_entropies[game_id] = initial_entropy
                bt.logging.debug(
                    f"Initialized entropy for game {game_id}: {initial_entropy}"
                )

        # Update entropy scores based on predictions
        for i, miner_predictions in enumerate(predictions):
            if miner_predictions.numel() > 0:
                miner_entropy = 0
                valid_predictions = 0
                for pred in miner_predictions:
                    game_id, outcome, odds, wager = pred
                    game_id = int(game_id.item())
                    if game_id < closing_line_odds.shape[0]:
                        new_entropy = self.update_game_entropy(
                            game_id, outcome, odds, wager
                        )
                        miner_entropy += new_entropy
                        valid_predictions += 1
                        bt.logging.debug(
                            f"Miner {i}, Game {game_id}: New entropy {new_entropy}"
                        )

                if valid_predictions > 0:
                    entropy_scores[i] = miner_entropy / valid_predictions
                else:
                    entropy_scores[i] = self.epsilon
            else:
                bt.logging.debug(f"Miner {i} has no predictions")

        # Ensure non-zero scores
        entropy_scores = t.clamp(entropy_scores, min=self.epsilon)

        # Normalize scores using softmax instead of min-max normalization
        entropy_scores = t.softmax(entropy_scores, dim=0)

        # Update the current time slot with new scores
        self.ebdr_scores[:, self.current_day, self.current_hour] = entropy_scores

        self._increment_time()

        return self.ebdr_scores

    def calculate_initial_entropy(self, odds):
        """
        Calculate the initial entropy for a game based on the odds.

        Args:
            odds (torch.Tensor): Tensor of odds for the game.

        Returns:
            float: Initial entropy value.
        """
        probs = self.calculate_bookmaker_probabilities(odds[1:])  # Exclude game_id
        entropy = -t.sum(probs * t.log2(probs + self.epsilon))
        result = max(entropy.item(), self.epsilon)
        bt.logging.debug(f"Initial entropy calculation: odds={odds}, result={result}")
        return result

    def update_game_entropy(self, game_id, outcome, odds, wager):
        """
        Update the entropy for a specific game based on new predictions.

        Args:
            game_id (int): ID of the game.
            outcome (float): Outcome of the game.
            odds (float): Odds for the game.
            wager (float): Wager amount.

        Returns:
            float: Updated entropy value.
        """
        current_entropy = self.game_entropies[game_id]
        prob = 1 / (odds + self.epsilon)
        new_entropy = -prob * math.log2(prob + self.epsilon)
        updated_entropy = max((current_entropy + new_entropy) / 2, self.epsilon)
        self.game_entropies[game_id] = updated_entropy
        bt.logging.debug(
            f"Updating game {game_id} entropy: current={current_entropy}, new={new_entropy}, updated={updated_entropy}"
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
        Get the current EBDR scores for the current time slot.

        Returns:
            torch.Tensor: Current EBDR scores.
        """
        return self.ebdr_scores[:, self.current_day, self.current_hour]

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

    def _increment_time(self):
        """
        Increment the current time slot by one hour. If the hour exceeds the scores per day, increment the day.
        """
        self.current_hour = (self.current_hour + 1) % self.scores_per_day
        if self.current_hour == 0:
            self.current_day = (self.current_day + 1) % self.max_days

    def get_current_entropy_scores(self):
        """
        Get the current entropy scores for the current time slot.

        Returns:
            torch.Tensor: Current entropy scores.
        """
        return self.ebdr_scores[:, self.current_day, self.current_hour]
