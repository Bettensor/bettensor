"""
BetTensor Scoring Module. 

This module handles the scoring of miners based on their data. The scoring is intended to select for miners that deploy positive expected value strategies, with some degree of risk management. 
We mostly determine +EV through closing line value analysis.

Inputs: 
- Miner Predictions

Outputs: 
- A torch array of the composite scores for all miners, indexed by miner_uid. 
"""


import torch as t
import logging
import bittensor as bt
from datetime import datetime, timezone
from typing import List, Dict
from .scoring_data import ScoringData
from .entropy_system import EntropySystem


class ScoringSystem:
    def __init__(
        self,
        db_path: str,
        num_miners: int,
        max_days: int,
        num_tiers: int = 5,
        reference_date: datetime = datetime(2023, 1, 1, tzinfo=timezone.utc),
    ):
        """
        Initialize the ScoringSystem.

        Args:
            db_path (str): Path to the database.
            num_miners (int): Number of miners.
            max_days (int): Maximum number of days to track.
            num_tiers (int, optional): Number of tiers. Defaults to 5.
            reference_date (datetime, optional): Reference date for scoring. Defaults to January 1, 2023.
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.num_miners = num_miners
        self.max_days = max_days
        self.num_tiers = num_tiers

        self.reference_date = reference_date

        # Initialize tier configurations
        self.tier_configs = [
            {
                "window": 3,
                "min_wager": 0,
                "capacity": int(num_miners * 1.0),
                "incentive": 0.1,
            },
            {
                "window": 7,
                "min_wager": 4000,
                "capacity": int(num_miners * 0.2),
                "incentive": 0.15,
            },
            {
                "window": 15,
                "min_wager": 10000,
                "capacity": int(num_miners * 0.2),
                "incentive": 0.2,
            },
            {
                "window": 30,
                "min_wager": 20000,
                "capacity": int(num_miners * 0.1),
                "incentive": 0.25,
            },
            {
                "window": 45,
                "min_wager": 35000,
                "capacity": int(num_miners * 0.05),
                "incentive": 0.3,
            },
        ]

        # Initialize score tensors
        self.clv_scores = t.zeros(num_miners, max_days)
        self.roi_scores = t.zeros(num_miners, max_days)
        self.amount_wagered = t.zeros(num_miners, max_days)
        self.entropy_scores = t.zeros(num_miners, max_days)
        self.tiers = t.ones(num_miners, max_days, dtype=t.int)

        # Initialize windowed score tensors
        self.window_clv_scores = t.zeros(num_miners, max_days, len(self.tier_configs))
        self.window_roi_scores = t.zeros(num_miners, max_days, len(self.tier_configs))
        self.window_sortino_scores = t.zeros(
            num_miners, max_days, len(self.tier_configs)
        )

        # Initialize composite scores
        self.composite_scores = t.zeros(num_miners, max_days, len(self.tier_configs))

        # Scoring weights
        self.clv_weight = 0.30
        self.roi_weight = 0.30
        self.ssi_weight = 0.30
        self.entropy_weight = 0.10
        self.entropy_window = 30

        self.scoring_data = ScoringData(db_path, num_miners)
        self.entropy_system = EntropySystem(num_miners, max_days)

        self.current_day = 0

    def update_scores(self, predictions, closing_line_odds, results):
        """
        Update the scores for the current day based on predictions, closing line odds, and results.

        Args:
            predictions (List[torch.Tensor]): List of miner predictions.
            closing_line_odds (torch.Tensor): Tensor of closing line odds.
            results (torch.Tensor): Tensor of game results.
        """
        self.logger.info(f"Updating scores for day {self.current_day}")
        try:
            self._update_raw_scores(predictions, closing_line_odds, results)
            self._update_window_scores()
            self._update_composite_scores()

            self.log_score_summary()
        except Exception as e:
            self.logger.error(f"Error updating scores: {str(e)}")
            raise

    def _update_raw_scores(
        self,
        predictions: List[t.Tensor],
        closing_line_odds: t.Tensor,
        results: t.Tensor,
    ):
        """
        Update the raw scores (CLV, ROI, and amount wagered) for the current day.

        Args:
            predictions (List[torch.Tensor]): List of miner predictions.
            closing_line_odds (torch.Tensor): Tensor of closing line odds.
            results (torch.Tensor): Tensor of game results.
        """
        clv_scores = t.zeros(self.num_miners)
        roi_scores = t.zeros(self.num_miners)
        wagers = t.zeros(self.num_miners)

        for i, miner_predictions in enumerate(predictions):
            if miner_predictions.size(0) > 0:
                game_ids = miner_predictions[:, 0].long()
                predicted_outcomes = miner_predictions[:, 1]
                predicted_odds = miner_predictions[:, 2]
                wager_amounts = miner_predictions[:, 3]

                # CLV calculation
                game_indices = game_ids.clamp(max=closing_line_odds.shape[0] - 1)
                relevant_closing_odds = closing_line_odds[
                    game_indices, predicted_outcomes.long() + 1
                ]
                clv = (predicted_odds - relevant_closing_odds) / relevant_closing_odds
                clv_scores[i] = clv.mean()

                # ROI calculation
                prediction_results = results[game_indices]
                returns = t.where(
                    prediction_results == predicted_outcomes,
                    (predicted_odds - 1) * wager_amounts,
                    -wager_amounts,
                )
                roi = returns / wager_amounts
                roi_scores[i] = roi.mean()

                wagers[i] = wager_amounts.sum()

        self._set_tensor_for_day(self.clv_scores, self.current_day, clv_scores)
        self._set_tensor_for_day(self.roi_scores, self.current_day, roi_scores)
        self._set_tensor_for_day(self.amount_wagered, self.current_day, wagers)

        # Update entropy scores
        self.entropy_system.update_ebdr_scores(predictions, closing_line_odds, results)
        entropy_scores = self.entropy_system.get_current_entropy_scores()
        self._set_tensor_for_day(self.entropy_scores, self.current_day, entropy_scores)

        self.logger.info(
            f"Entropy scores - min: {entropy_scores.min().item():.8f}, "
            f"max: {entropy_scores.max().item():.8f}, "
            f"mean: {entropy_scores.mean().item():.8f}, "
            f"non-zero: {(entropy_scores != 0).sum().item()}"
        )

    def _update_window_scores(self):
        """
        Update the windowed scores (CLV, ROI, and Sortino ratio) for the current day.
        """
        for tier, config in enumerate(self.tier_configs):
            window = min(config["window"], self.max_days)

            start_day = self.current_day

            clv_window = self._get_window_scores(self.clv_scores, start_day, window)
            roi_window = self._get_window_scores(self.roi_scores, start_day, window)

            # Take the mean over the window
            self._set_tensor_for_day(
                self.window_clv_scores, self.current_day, clv_window.mean(dim=1), tier
            )
            self._set_tensor_for_day(
                self.window_roi_scores, self.current_day, roi_window.mean(dim=1), tier
            )

            # Calculate Sortino ratio
            target_return = 0  # You can adjust this if needed
            excess_returns = roi_window - target_return
            downside_returns = t.where(
                excess_returns < 0, excess_returns, t.zeros_like(excess_returns)
            )
            expected_return = excess_returns.mean(dim=1)
            downside_deviation = t.sqrt((downside_returns**2).mean(dim=1))
            sortino = expected_return / (downside_deviation + 1e-8)
            sortino = t.clamp(
                sortino, min=-10, max=10
            )  # Cap Sortino ratio between -10 and 10
            self._set_tensor_for_day(
                self.window_sortino_scores, self.current_day, sortino, tier
            )

            self.logger.info(
                f"Tier {tier} Sortino ratio - min: {sortino.min().item():.4f}, "
                f"max: {sortino.max().item():.4f}, "
                f"mean: {sortino.mean().item():.4f}"
            )

    def _update_composite_scores(self):
        """
        Update the composite scores for the current day based on the windowed scores.
        """
        for tier in range(len(self.tier_configs)):
            clv = self._get_tensor_for_day(
                self.window_clv_scores, self.current_day, tier
            )
            roi = self._get_tensor_for_day(
                self.window_roi_scores, self.current_day, tier
            )
            sortino = self._get_tensor_for_day(
                self.window_sortino_scores, self.current_day, tier
            )
            entropy = self._get_tensor_for_day(self.entropy_scores, self.current_day)

            # Normalize each component score
            clv_norm = (clv - clv.mean()) / (clv.std() + 1e-8)
            roi_norm = (roi - roi.mean()) / (roi.std() + 1e-8)
            sortino_norm = (sortino - sortino.mean()) / (sortino.std() + 1e-8)
            entropy_norm = (entropy - entropy.mean()) / (entropy.std() + 1e-8)

            composite_score = (
                self.clv_weight * clv_norm
                + self.roi_weight * roi_norm
                + self.ssi_weight * sortino_norm
                + self.entropy_weight * entropy_norm
            )

            # Normalize the composite score
            composite_score = (composite_score - composite_score.mean()) / (
                composite_score.std() + 1e-8
            )

            self._set_tensor_for_day(
                self.composite_scores, self.current_day, composite_score, tier
            )

    def _get_window_scores(self, score_tensor, start_day, window):
        """
        Get the windowed scores for a given tensor and window size.

        Args:
            score_tensor (torch.Tensor): The tensor containing the scores.
            start_day (int): The starting day for the window.
            window (int): The size of the window.

        Returns:
            torch.Tensor: The windowed scores.
        """
        scores = []
        for i in range(window):
            day = (start_day - i + self.max_days) % self.max_days
            scores.append(self._get_tensor_for_day(score_tensor, day))
        return t.stack(
            scores, dim=1
        )  # Stack along dimension 1 to preserve miner dimension

    def _increment_time(self, new_day):
        """
        Increment the current day to a new day, updating the tiers accordingly.

        Args:
            new_day (int): The new day to set.
        """
        if new_day != self.current_day:
            target_day = self._get_day_index(new_day)
            source_day = self._get_day_index(self.current_day)
            self.tiers[:, target_day] = self.tiers[:, source_day]

        self.current_day = new_day % self.max_days

    def set_current_time(self, day):
        """
        Set the current day to a specific day.

        Args:
            day (int): The day to set as the current day.
        """
        self.current_day = day % self.max_days

    def calculate_composite_scores(self):
        """
        Calculate the composite scores for all miners based on their windowed scores.

        Returns:
            torch.Tensor: The composite scores for all miners.
        """
        composite_scores = t.zeros(
            (self.num_miners, self.max_days, len(self.tier_configs))
        )

        for tier, config in enumerate(self.tier_configs):
            window = min(config["window"], self.max_days)

            start_day = self.current_day

            clv = self._get_window_scores(self.clv_scores, start_day, window)
            roi = self._get_window_scores(self.roi_scores, start_day, window)
            sortino = self._get_window_scores(
                self.window_sortino_scores[:, :, tier], start_day, window
            )
            entropy = self._get_window_scores(self.entropy_scores, start_day, window)

            if (
                clv.numel() > 0
                and roi.numel() > 0
                and sortino.numel() > 0
                and entropy.numel() > 0
            ):
                composite_score = (
                    self.clv_weight * clv.mean(dim=1)
                    + self.roi_weight * roi.mean(dim=1)
                    + self.ssi_weight * sortino.mean(dim=1)
                    + self.entropy_weight * entropy.mean(dim=1)
                )
                self._set_tensor_for_day(
                    composite_scores, self.current_day, composite_score, tier
                )

        return composite_scores

    def manage_tiers(self):
        """
        Manage the tiers of miners based on their composite scores.
        """
        self.logger.info("Managing tiers")

        try:
            composite_scores = self.calculate_composite_scores()
            current_tiers = self._get_tensor_for_day(self.tiers, self.current_day)
            new_tiers = current_tiers.clone()

            initial_miner_count = (current_tiers > 0).sum()

            # Step 1: Handle demotions
            self._handle_demotions(new_tiers, composite_scores)

            # Step 2: Handle promotions, swaps, and fill empty slots
            self._handle_promotions_and_fill_slots(new_tiers, composite_scores)

            # Step 3: Force another round of slot filling
            self._fill_all_empty_slots(new_tiers, composite_scores)

            # Ensure that all tier indices are within bounds
            assert (
                new_tiers.max() < self.num_tiers
            ), f"Tier index {new_tiers.max()} exceeds configured tiers."

            # Update tiers for the current day and hour
            self._set_tensor_for_day(self.tiers, self.current_day, new_tiers)

            # Propagate the new tier information to the next day
            next_day = (self.current_day + 1) % self.max_days
            self.tiers[:, next_day] = new_tiers

            final_miner_count = (new_tiers > 0).sum()
            assert (
                initial_miner_count == final_miner_count
            ), f"Miner count changed from {initial_miner_count} to {final_miner_count}"

            self.logger.info("Tier management completed")
            self.log_tier_summary()

        except Exception as e:
            self.logger.error(f"Error managing tiers: {str(e)}")
            raise

    def _handle_demotions(self, tiers, composite_scores):
        """
        Handle the demotion of miners based on their composite scores and tier configurations.

        Args:
            tiers (torch.Tensor): The tensor containing the current tiers of miners.
            composite_scores (torch.Tensor): The tensor containing the composite scores of miners.
        """
        initial_tiers = tiers.clone()
        for tier in range(len(self.tier_configs), 1, -1):
            config = self.tier_configs[tier - 1]
            tier_mask = tiers == tier
            tier_miners = tier_mask.nonzero().squeeze()

            # Check minimum wager requirement
            wager = self._get_cumulative_wager(tier_miners, config["window"])
            demotion_mask = wager < config["min_wager"]

            demoted_miners = tier_miners[demotion_mask]
            for miner in demoted_miners:
                self._cascade_demotion(miner, tier, tiers, composite_scores)
                self.logger.info(
                    f"Miner {miner.item()} demoted from tier {tier} due to insufficient wager: {wager[miner == tier_miners].item():.2f} < {config['min_wager']}"
                )
        self.logger.info(f"Demotions made: {(initial_tiers != tiers).sum().item()}")

    def _handle_promotions_and_fill_slots(self, tiers, composite_scores):
        """
        Handle the promotion and filling of empty slots for miners based on their composite scores and tier configurations.

        Args:
            tiers (torch.Tensor): The tensor containing the current tiers of miners.
            composite_scores (torch.Tensor): The tensor containing the composite scores of miners.
        """
        initial_tiers = tiers.clone()
        for tier in range(1, len(self.tier_configs)):
            config = self.tier_configs[tier - 1]
            next_config = self.tier_configs[tier]

            # Handle promotions and swaps
            self._promote_and_swap(tier, tiers, composite_scores, config, next_config)

            # Fill empty slots immediately after promotions
            self._fill_empty_slots(tier, tiers, composite_scores, config)

        # Repeat the fill process to ensure all slots are filled
        for tier in range(1, len(self.tier_configs)):
            config = self.tier_configs[tier - 1]
            self._fill_empty_slots(tier, tiers, composite_scores, config)

        self.logger.info(
            f"Promotions and fills made: {(initial_tiers != tiers).sum().item()}"
        )

    def _promote_and_swap(self, tier, tiers, composite_scores, config, next_config):
        """
        Promote and swap miners between tiers based on their composite scores and tier configurations.

        Args:
            tier (int): The current tier.
            tiers (torch.Tensor): The tensor containing the current tiers of miners.
            composite_scores (torch.Tensor): The tensor containing the composite scores of miners.
            config (dict): The configuration of the current tier.
            next_config (dict): The configuration of the next tier.
        """
        tier_mask = tiers == tier
        tier_miners = tier_mask.nonzero().squeeze()

        if tier_miners.numel() == 0:
            return

        # Check minimum wager requirement for promotion
        wager = self._get_cumulative_wager(tier_miners, next_config["window"])
        promotion_eligible = wager >= next_config["min_wager"]

        if not promotion_eligible.any():
            return

        # Get scores for eligible miners
        eligible_miners = tier_miners[promotion_eligible]
        eligible_scores = composite_scores[eligible_miners, self.current_day, tier - 1]

        # Check if miners are in top 50% of their current tier
        median_score = eligible_scores.median()
        top_half_mask = eligible_scores >= median_score
        top_half_miners = eligible_miners[top_half_mask]
        top_half_scores = eligible_scores[top_half_mask]

        if top_half_miners.numel() == 0:
            return

        # Check next tier capacity
        next_tier_mask = tiers == (tier + 1)
        next_tier_miners = next_tier_mask.nonzero().squeeze()
        available_slots = max(0, next_config["capacity"] - next_tier_miners.numel())

        if available_slots > 0:
            # Promote miners to fill available slots
            promotions = min(available_slots, top_half_miners.numel())
            promoted_miners = top_half_miners[:promotions]
            tiers[promoted_miners] = tier + 1
            for miner in promoted_miners:
                self.logger.info(f"Miner {miner.item()} promoted to tier {tier + 1}")

        # Handle swaps if next tier is at capacity
        if next_tier_miners.numel() == next_config["capacity"]:
            next_tier_scores = composite_scores[
                next_tier_miners, self.current_day, tier
            ]
            for i, (miner, score) in enumerate(zip(top_half_miners, top_half_scores)):
                if i >= next_tier_miners.numel():
                    break
                if score > next_tier_scores[i]:
                    # Swap miners
                    tiers[miner] = tier + 1
                    tiers[next_tier_miners[i]] = tier
                    self.logger.info(
                        f"Miner {miner.item()} promoted to tier {tier + 1}, replacing miner {next_tier_miners[i].item()}"
                    )
                else:
                    # Stop swapping if we reach a higher score in the next tier
                    break

    def _fill_empty_slots(self, tier, tiers, composite_scores, config):
        """
        Fill empty slots in a tier by promoting miners from the lower tier based on their composite scores.

        Args:
            tier (int): The current tier.
            tiers (torch.Tensor): The tensor containing the current tiers of miners.
            composite_scores (torch.Tensor): The tensor containing the composite scores of miners.
            config (dict): The configuration of the current tier.
        """
        tier_mask = tiers == tier
        tier_miners = tier_mask.nonzero().squeeze()

        empty_slots = max(0, config["capacity"] - tier_miners.numel())

        if empty_slots == 0:
            return

        lower_tier_mask = tiers == (tier - 1)
        lower_tier_miners = lower_tier_mask.nonzero().squeeze()

        if lower_tier_miners.numel() == 0:
            return

        # Check minimum wager requirement for promotion
        wager = self._get_cumulative_wager(lower_tier_miners, config["window"])
        promotion_eligible = wager >= config["min_wager"]

        if not promotion_eligible.any():
            return

        # Get scores for eligible miners
        eligible_miners = lower_tier_miners[promotion_eligible]
        eligible_scores = composite_scores[eligible_miners, self.current_day, tier - 2]

        # Sort eligible miners by score
        sorted_indices = eligible_scores.argsort(descending=True)
        sorted_miners = eligible_miners[sorted_indices]

        # Promote miners to fill empty slots
        promotions = min(empty_slots, sorted_miners.numel())
        promoted_miners = sorted_miners[:promotions]
        tiers[promoted_miners] = tier
        for miner in promoted_miners:
            self.logger.info(
                f"Miner {miner.item()} promoted to tier {tier} to fill empty slot"
            )

    def _fill_all_empty_slots(self, tiers, composite_scores):
        """
        Fill all empty slots in all tiers by promoting miners from lower tiers based on their composite scores.

        Args:
            tiers (torch.Tensor): The tensor containing the current tiers of miners.
            composite_scores (torch.Tensor): The tensor containing the composite scores of miners.
        """
        for tier in range(1, len(self.tier_configs)):
            config = self.tier_configs[tier - 1]
            self._fill_empty_slots(tier, tiers, composite_scores, config)

    def _get_cumulative_wager(self, miners, window):
        """
        Get the cumulative wager for a set of miners over a specified window.

        Args:
            miners (torch.Tensor): The tensor containing the indices of the miners.
            window (int): The size of the window.

        Returns:
            torch.Tensor: The cumulative wager for the miners.
        """
        end_day = self.current_day
        start_day = max(0, end_day - window + 1)

        if start_day <= end_day:
            wager = self.amount_wagered[miners, start_day : end_day + 1].sum(dim=1)
        else:
            wager = self.amount_wagered[miners, start_day:].sum(
                dim=1
            ) + self.amount_wagered[miners, : end_day + 1].sum(dim=1)

        return wager

    def _cascade_demotion(self, miner, current_tier, tiers, composite_scores):
        """
        Cascade the demotion of a miner through lower tiers based on their composite scores.

        Args:
            miner (int): The index of the miner to demote.
            current_tier (int): The current tier of the miner.
            tiers (torch.Tensor): The tensor containing the current tiers of miners.
            composite_scores (torch.Tensor): The tensor containing the composite scores of miners.
        """
        for lower_tier in range(current_tier - 1, 0, -1):
            config = self.tier_configs[lower_tier - 1]
            lower_tier_mask = tiers == lower_tier
            lower_tier_miners = lower_tier_mask.nonzero().squeeze()

            if lower_tier_miners.numel() < config["capacity"]:
                # There's space in this tier, demote the miner here
                tiers[miner] = lower_tier
                break
            else:
                # Compare scores
                miner_score = composite_scores[miner, self.current_day, lower_tier - 1]
                lower_tier_scores = composite_scores[
                    lower_tier_miners, self.current_day, lower_tier - 1
                ]
                if miner_score > lower_tier_scores.min():
                    # Swap with the lowest scoring miner in this tier
                    lowest_miner = lower_tier_miners[lower_tier_scores.argmin()]
                    tiers[miner] = lower_tier
                    self._cascade_demotion(
                        lowest_miner, lower_tier, tiers, composite_scores
                    )
                    break
        else:
            # If we've reached here, the miner goes to tier 1
            tiers[miner] = 1

    def reset_miner(self, miner_uid: int):
        """
        Initialize a new miner, replacing any existing miner at the same UID.

        Args:
            miner_uid (int): The UID of the miner to reset.
        """
        self.clv_scores[miner_uid] = 0
        self.roi_scores[miner_uid] = 0
        self.amount_wagered[miner_uid] = 0
        self.composite_scores[miner_uid] = 0
        self.entropy_scores[miner_uid] = 0
        self.tiers[miner_uid] = 1

    def log_tier_summary(self, message="Current tier distribution"):
        """
        Log the summary of tier distributions.

        Args:
            message (str, optional): The message to log before the summary. Defaults to "Current tier distribution".
        """
        self.logger.info(message)
        current_tiers = self._get_tensor_for_day(self.tiers, self.current_day)
        for tier in range(1, len(self.tier_configs) + 1):
            tier_count = (current_tiers == tier).sum().item()
            self.logger.info(f"Tier {tier}: {tier_count} miners")

    def get_miner_history(self, miner_uid: int, score_type: str, days: int = None):
        """
        Get the score history for a specific miner and score type.

        Args:
            miner_uid (int): The UID of the miner.
            score_type (str): The type of score to retrieve ('clv', 'roi', 'sortino', 'entropy', 'composite', 'tier').
            days (int, optional): Number of days of history to return. If None, returns all available history.

        Returns:
            torch.Tensor: A tensor containing the miner's score history.
        """
        score_tensor = getattr(self, f"{score_type}_scores", None)
        if score_tensor is None:
            raise ValueError(f"Invalid score type: {score_type}")

        if days is None:
            return score_tensor[miner_uid]
        else:
            return score_tensor[miner_uid, -days:]

    def log_score_summary(self):
        """
        Log a summary of the current scores.
        """
        self.logger.info("=== Score Summary ===")
        for score_name, score_tensor in [
            ("CLV", self.window_clv_scores),
            ("ROI", self.window_roi_scores),
            ("Sortino", self.window_sortino_scores),
            ("Entropy", self.entropy_scores),
            ("Composite", self.composite_scores),
        ]:
            current_scores = self._get_tensor_for_day(score_tensor, self.current_day)
            self.logger.info(
                f"{score_name} Scores - min: {current_scores.min().item():.4f}, "
                f"max: {current_scores.max().item():.4f}, "
                f"mean: {current_scores.mean().item():.4f}, "
                f"non-zero: {(current_scores != 0).sum().item()}"
            )

    def calculate_weights(self):
        """
        Calculate weights for all miners based on their tier and composite score.
        Weights sum to 1 and represent both the miner's share of incentives and their influence.

        Returns:
            torch.Tensor: The calculated weights for all miners.
        """
        self.logger.info("Calculating weights")

        try:
            weights = t.zeros(self.num_miners)

            tier_incentives = t.tensor(
                [config["incentive"] for config in self.tier_configs]
            )
            total_incentive = tier_incentives.sum()
            normalized_incentives = tier_incentives / total_incentive

            # Calculate the number of miners in each tier
            current_tiers = self._get_tensor_for_day(self.tiers, self.current_day)
            tier_counts = t.bincount(
                current_tiers, minlength=len(self.tier_configs) + 1
            )[1:]

            # Only consider non-empty tiers
            non_empty_tiers = tier_counts > 0
            active_tiers = t.arange(1, len(self.tier_configs) + 1)[non_empty_tiers]

            # Redistribute weights from empty tiers
            total_active_incentive = normalized_incentives[non_empty_tiers].sum()
            adjusted_incentives = (
                normalized_incentives[non_empty_tiers] / total_active_incentive
            )

            for i, tier in enumerate(active_tiers):
                tier_mask = current_tiers == tier
                tier_scores = self._get_tensor_for_day(
                    self.composite_scores, self.current_day, tier - 1
                )[tier_mask]

                if tier_scores.numel() == 0:
                    continue

                # Normalize scores within the tier
                tier_weights = t.softmax(tier_scores, dim=0)

                # Apply adjusted tier incentive
                tier_weights *= adjusted_incentives[i]

                weights[tier_mask] = tier_weights

            # Ensure weights sum to 1
            total_weight = weights.sum()
            if total_weight > 0:
                weights /= total_weight
            else:
                weights = t.ones(self.num_miners) / self.num_miners

            self.logger.info(
                f"Weights calculated - min: {weights.min().item():.4f}, max: {weights.max().item():.4f}, mean: {weights.mean().item():.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error calculating weights: {str(e)}")
            raise

        return weights

    def scoring_run(self, current_date):
        """
        Perform a scoring run for the given date.

        Args:
            current_date (datetime): The date for which to perform the scoring run.

        Returns:
            torch.Tensor: The calculated weights for all miners.
        """
        self.logger.debug(f"Starting scoring run for date: {current_date}")

        # Ensure current_date is timezone-aware
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=timezone.utc)

        date_str = current_date.strftime("%Y-%m-%d")
        self.logger.info(f"=== Starting scoring run for date: {date_str} ===")

        (
            predictions,
            closing_line_odds,
            results,
        ) = self.scoring_data.preprocess_for_scoring(date_str)

        self.logger.info(f"Number of predictions: {len(predictions)}")
        self.logger.info(f"Closing line odds shape: {closing_line_odds.shape}")
        self.logger.info(f"Results shape: {results.shape}")

        # Calculate the days since reference date without wraparound
        days_since_reference = (current_date - self.reference_date).days
        new_day = days_since_reference % self.max_days

        if new_day != self.current_day:
            self.logger.info(f"Moving from day {self.current_day} to day {new_day}")
            self._increment_time(new_day)

        # Update the current_date
        self.current_date = current_date

        self.logger.info(
            f"Current day: {self.current_day}, reference date: {self.reference_date}"
        )

        # Log initial tier distribution
        self.log_tier_summary("Initial tier distribution")

        # Add this debugging code before calculating Sortino ratios
        current_tiers = self._get_tensor_for_day(self.tiers, self.current_day)
        tier_distribution = [
            int((current_tiers == tier).sum().item())
            for tier in range(1, len(self.tier_configs) + 1)
        ]
        self.logger.info(f"Current tier distribution: {tier_distribution}")

        if predictions:
            self.logger.info("Updating scores...")
            self.update_scores(predictions, closing_line_odds, results)
            self.logger.info("Scores updated successfully.")

            total_wager = (
                self._get_tensor_for_day(self.amount_wagered, self.current_day)
                .sum()
                .item()
            )
            avg_wager = total_wager / self.num_miners
            self.logger.info(f"Total wager for this run: {total_wager:.2f}")
            self.logger.info(f"Average wager per miner: {avg_wager:.2f}")
        else:
            self.logger.warning(
                f"No predictions for date {date_str}. Skipping score update."
            )

        self.logger.info("Calculating weights...")
        weights = self.calculate_weights()
        self.logger.info(
            f"Weights calculated. Min: {weights.min().item():.4f}, Max: {weights.max().item():.4f}, Mean: {weights.mean().item():.4f}"
        )

        self.logger.info("Managing tiers...")
        self.manage_tiers()
        self.logger.info("Tiers managed successfully.")

        # Log final tier distribution
        self.log_tier_summary("Final tier distribution")

        self.log_score_summary()

        self.logger.info(f"=== Completed scoring run for date: {date_str} ===")

        return weights

    def reset_date(self, new_date):
        """
        Reset the scoring system's date for testing purposes.

        Args:
            new_date (datetime): The new date to set as the current date.
        """
        # Ensure new_date is timezone-aware
        if new_date.tzinfo is None:
            new_date = new_date.replace(tzinfo=timezone.utc)

        self.current_date = new_date
        days_since_reference = (new_date - self.reference_date).days
        self.current_day = days_since_reference % self.max_days

        # Reset tiers to 1 for the current day, using modulo for wraparound
        self._set_tensor_for_day(
            self.tiers, self.current_day, t.ones(self.num_miners, dtype=t.int)
        )

    def reset_all_miners_to_tier_1(self):
        """
        Reset all miners to tier 1.
        """
        self.tiers.fill_(1)

    def _get_day_index(self, day):
        """
        Get the index of a day in the circular buffer.

        Args:
            day (int): The day to get the index for.

        Returns:
            int: The index of the day in the circular buffer.
        """
        return day % self.max_days

    def _get_tensor_for_day(self, tensor, day, tier=None):
        """
        Get the tensor for a specific day and tier.

        Args:
            tensor (torch.Tensor): The tensor to get the data from.
            day (int): The day to get the data for.
            tier (int, optional): The tier to get the data for. Defaults to None.

        Returns:
            torch.Tensor: The tensor for the specified day and tier.
        """
        if tier is None:
            return tensor[:, self._get_day_index(day)]
        else:
            return tensor[:, self._get_day_index(day), tier]

    def _set_tensor_for_day(self, tensor, day, value, tier=None):
        """
        Set the tensor for a specific day and tier.

        Args:
            tensor (torch.Tensor): The tensor to set the data in.
            day (int): The day to set the data for.
            value (torch.Tensor): The value to set in the tensor.
            tier (int, optional): The tier to set the data for. Defaults to None.
        """
        if tier is None:
            tensor[:, self._get_day_index(day)] = value
        else:
            tensor[:, self._get_day_index(day), tier] = value

    def calculate_sortino_ratio(self, tier):
        # Add this check at the beginning of the method
        if (self.tiers == tier).sum() == 0:
            self.logger.info(f"No miners in Tier {tier}, skipping Sortino ratio calculation")
            return None

        # ... rest of the method ...

    def update_tiers(self):
        # Add logging here to track tier changes
        old_tiers = self._get_tensor_for_day(self.tiers, self.current_day).clone()
        
        # ... existing tier update logic ...

        new_tiers = self._get_tensor_for_day(self.tiers, self.current_day)
        changes = (old_tiers != new_tiers).sum().item()
        self.logger.info(f"Updated tiers: {changes} miners changed tiers")
