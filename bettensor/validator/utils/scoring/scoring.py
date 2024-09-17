'''
BetTensor Scoring Module. 

This module handles the scoring of miners based on their data. The scoring is intended to select for miners that deploy positive expected value strategies, with some degree of risk management. 
We mostly determine +EV through closing line value analysis.

Inputs: 
- Miner Predictions

Outputs: 
- A torch array of the composite scores for all miners, indexed by miner_uid. 
'''



import numpy as np
import torch as t
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict
from .scoring_data import ScoringData
from .entropy_system import EntropySystem

class ScoringSystem:
    def __init__(self, db_path, num_miners=256, max_days=45):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.num_miners = num_miners
        self.max_days = max_days
        
        # Initialize all component score tensors with 2D structure
        self.clv_scores = t.zeros(num_miners, max_days)
        self.sortino_scores = t.zeros(num_miners, max_days)
        self.roi_scores = t.zeros(num_miners, max_days)
        self.entropy_scores = t.zeros(num_miners, max_days)
        self.composite_scores = t.zeros(num_miners, max_days)
        
        self.amount_wagered = t.zeros(num_miners, max_days)
        self.tiers = t.ones(num_miners, dtype=t.int)
        self.tier_history = t.ones(num_miners, max_days, dtype=t.int)

        # WARNING: DO NOT CHANGE THESE VALUES. THEY WILL IMPACT VTRUST SIGNIFICANTLY. 
        self.clv_weight = 0.30
        self.roi_weight = 0.30
        self.ssi_weight = 0.30
        self.entropy_weight = 0.10
        self.entropy_window = 30

        # Update tier configurations
        self.tier_configs = [
            {'window': 3, 'min_wager': 0, 'capacity': int(num_miners * 1.0), 'incentive': 0.1},
            {'window': 7, 'min_wager': 4000, 'capacity': int(num_miners * 0.2), 'incentive': 0.15},
            {'window': 15, 'min_wager': 10000, 'capacity': int(num_miners * 0.2), 'incentive': 0.2},
            {'window': 30, 'min_wager': 20000, 'capacity': int(num_miners * 0.1), 'incentive': 0.25},
            {'window': 45, 'min_wager': 35000, 'capacity': int(num_miners * 0.05), 'incentive': 0.3}
        ]

        self.daily_predictions = [[] for _ in range(max_days)]  # List of lists to store daily predictions

        self.last_update_date = None  # Store the date of the last update
        self.current_date = None  # 

        self.scoring_data = ScoringData(db_path, num_miners)
        self.entropy_system = EntropySystem(num_miners, max_days)

    def _get_current_date(self):
        """Get the current date in UTC."""
        return datetime.now(timezone.utc).date()

    def _should_shift(self, current_date):
        """Check if we should shift the tensors based on the current date."""
        if self.last_update_date is None:
            return False
        return current_date > self.last_update_date

    def _shift_tensors(self):
        """Shift all score tensors to the left by one position."""
        self.clv_scores = t.roll(self.clv_scores, shifts=-1, dims=1)
        self.sortino_scores = t.roll(self.sortino_scores, shifts=-1, dims=1)
        self.roi_scores = t.roll(self.roi_scores, shifts=-1, dims=1)
        self.entropy_scores = t.roll(self.entropy_scores, shifts=-1, dims=1)
        self.composite_scores = t.roll(self.composite_scores, shifts=-1, dims=1)
        self.amount_wagered = t.roll(self.amount_wagered, shifts=-1, dims=1)
        self.tier_history = t.roll(self.tier_history, shifts=-1, dims=1)

    def update_daily_scores(self, predictions, closing_line_odds, results):
        """
        Update daily scores based on predictions, closing line odds, and results.
        """
        self.logger.info("Updating daily scores")
        try:
            self.clv_scores = self._update_clv(predictions, closing_line_odds)
            self.roi_scores = self._update_roi(predictions, results)
            self.sortino_scores = self._update_sortino()
            self.entropy_scores = self.entropy_system.update_ebdr_scores(predictions, closing_line_odds, results)
            self.log_score_summary()
        except Exception as e:
            self.logger.error(f"Error updating daily scores: {str(e)}")
            raise

    def _validate_inputs(self, predictions: List[t.Tensor], closing_line_odds: t.Tensor, results: t.Tensor):
        if not isinstance(predictions, list) or not all(isinstance(p, t.Tensor) and p.dim() == 2 and p.size(1) == 3 for p in predictions):
            raise ValueError("predictions must be a list of 2D tensors with shape (num_predictions, 3)")
        if not isinstance(closing_line_odds, t.Tensor) or closing_line_odds.dim() != 2 or closing_line_odds.size(1) != 2:
            raise ValueError("closing_line_odds must be a 2D tensor with shape (num_games, 2)")
        if not isinstance(results, t.Tensor) or results.dim() != 1:
            raise ValueError("results must be a 1D tensor")

        print(f"Debug: Validation passed. Predictions: {len(predictions)}, Closing line odds: {closing_line_odds.shape}, Results: {results.shape}")

    def _update_clv(self, predictions: List[t.Tensor], closing_line_odds: t.Tensor) -> t.Tensor:
        clv_scores = t.zeros(self.num_miners, self.max_days)
        for i, pred in enumerate(predictions):
            game_ids, predicted_outcomes, predicted_odds = pred[:, 0], pred[:, 1], pred[:, 2]
            game_indices = game_ids.long().clamp(max=closing_line_odds.shape[0]-1)
            relevant_closing_odds = closing_line_odds[game_indices, 1]
            
            # Calculate CLV for each prediction
            clv = (predicted_odds - relevant_closing_odds) / relevant_closing_odds * 100
            clv_scores[i, -1] = clv.mean()
        return clv_scores

    def _update_roi(self, predictions: List[t.Tensor], results: t.Tensor) -> t.Tensor:
        roi_scores = t.zeros(self.num_miners)
        for i, miner_predictions in enumerate(predictions):
            if miner_predictions.size(0) > 0:
                game_ids = miner_predictions[:, 0].long()
                predicted_outcomes = miner_predictions[:, 1]
                submitted_odds = miner_predictions[:, 2]
                prediction_results = results[game_ids.clamp(max=results.shape[0]-1)]
                returns = t.where(prediction_results == predicted_outcomes, submitted_odds - 1, -1)
                roi_scores[i] = returns.mean()
        
        # Update ROI scores for the current day
        self.roi_scores[:, -1] = roi_scores
        
        return self.roi_scores

    def _update_sortino(self) -> t.Tensor:
        """
        Update Sortino scores based on ROI scores.
        """
        self.logger.info("Updating Sortino scores")
        try:
            negative_returns = t.where(self.roi_scores < 0, self.roi_scores, t.tensor(0.0))
            downside_deviation = t.sqrt((negative_returns ** 2).mean(dim=1))
            mean_returns = self.roi_scores.mean(dim=1)
            sortino_scores = mean_returns / (downside_deviation + 1e-8)

            # Update Sortino scores for the current day
            self.sortino_scores[:, -1] = sortino_scores

            return self.sortino_scores
        except Exception as e:
            self.logger.error(f"Error updating Sortino scores: {str(e)}")
            raise

    def _calculate_amount_wagered(self, predictions: List[t.Tensor]) -> t.Tensor:
        amount_wagered = t.zeros(self.num_miners)
        for i, p in enumerate(predictions):
            amount_wagered[i] = p.size(0)
        return amount_wagered

    def _update_entropy(self, predictions: List[t.Tensor]) -> t.Tensor:
        entropy_scores = t.zeros(self.num_miners, self.max_days)
        for i, pred in enumerate(predictions):
            probabilities = t.softmax(pred[:, 1:], dim=1)  # Convert odds to probabilities
            entropy = -t.sum(probabilities * t.log(probabilities), dim=1)
            entropy_scores[i, -1] = entropy.mean()
        return entropy_scores

    def _update_composite_scores(self) -> t.Tensor:
        composite_scores = (
            self.clv_scores.mean(dim=1) * self.clv_weight +
            self.roi_scores.mean(dim=1) * self.roi_weight +
            self.sortino_scores.mean(dim=1) * self.ssi_weight +
            self.entropy_scores.mean(dim=1) * self.entropy_weight
        )

        # Update composite scores for the current day
        self.composite_scores[:, -1] = composite_scores

        return self.composite_scores

    def calculate_composite_scores(self):
        """
        Calculate composite scores for all miners based on their current tier.
        """
        composite_scores = t.zeros(self.num_miners)
        for tier, config in enumerate(self.tier_configs, 1):
            tier_mask = self.tiers == tier
            window = min(config['window'], self.clv_scores.shape[1])  # Ensure window doesn't exceed available data
            
            # Calculate composite score for miners in this tier
            tier_scores = t.zeros(self.num_miners)
            tier_scores[tier_mask] = (
                self.clv_scores[tier_mask, -window:].mean(dim=1) +
                self.roi_scores[tier_mask, -window:].mean(dim=1) +
                self.sortino_scores[tier_mask, -window:].mean(dim=1) +
                self.entropy_scores[tier_mask, -window:].mean(dim=1)
            )
            
            composite_scores += tier_scores
        
        return composite_scores

    def reset_miner(self, miner_uid: int):
        """
        Initialize a new miner, replacing any existing miner at the same UID.
        """
        self.clv_scores[miner_uid] = 0
        self.sortino_scores[miner_uid] = 0
        self.roi_scores[miner_uid] = 0
        self.amount_wagered[miner_uid] = 0
        self.composite_scores[miner_uid] = 0
        self.entropy_scores[miner_uid] = 0
        
        self.tiers[miner_uid] = 1
        self.tier_history[miner_uid] = 1

    def calculate_composite_score(self, miner_uid: int, window: int) -> float:
        """
        Calculate the composite score for a miner over a given window.
        """
        scores = (
            self.clv_scores[miner_uid, -window:].mean() +
            self.roi_scores[miner_uid, -window:].mean() +
            self.sortino_scores[miner_uid, -window:].mean() +
            self.entropy_scores[miner_uid, -window:].mean() 
        )
        return scores.item()

    def manage_tiers(self, debug: bool = False):
        """
        Manage tier promotions and demotions based on defined rules.
        """
        self.logger.info("Managing tiers")

        try:
            old_tiers = self.tiers.clone()
            composite_scores = self.calculate_composite_scores()
            new_tiers = old_tiers.clone()

            # Demotions
            for miner in range(self.num_miners):
                current_tier = old_tiers[miner].item()
                if current_tier > 1:
                    window = self.tier_configs[current_tier - 1]['window']
                    min_wager = self.tier_configs[current_tier - 1]['min_wager']
                    wager = self.amount_wagered[miner, -window:].sum().item()

                    if wager < min_wager:
                        new_tiers[miner] = current_tier - 1
                        if debug:
                            self.logger.info(f"Miner {miner} demoted from tier {current_tier} to {current_tier - 1}")
                            self.logger.info(f"  Wager: {wager}, Min required: {min_wager}")

            # Promotions
            for tier in range(1, len(self.tier_configs)):
                current_tier_mask = new_tiers == tier
                next_tier = tier + 1
                next_tier_capacity = self.tier_configs[next_tier - 1]['capacity']
                next_tier_min_wager = self.tier_configs[next_tier - 1]['min_wager']
                next_tier_window = self.tier_configs[next_tier - 1]['window']

                # Calculate wager over the appropriate window for the next tier
                wager_over_window = self.amount_wagered[:, -next_tier_window:].sum(dim=1)

                # Identify eligible miners for promotion
                eligible_mask = (
                    (wager_over_window >= next_tier_min_wager) &
                    current_tier_mask &
                    (composite_scores > composite_scores[current_tier_mask].median())
                )

                eligible_miners = eligible_mask.nonzero(as_tuple=True)[0]
                if eligible_miners.numel() == 0:
                    continue

                # Check available slots in the next tier
                available_slots = next_tier_capacity - (new_tiers == next_tier).sum().item()
                if available_slots <= 0:
                    # Find miners to swap based on scores
                    lowest_next_tier_scores, lowest_indices = self.composite_scores[new_tiers == next_tier].topk(available_slots, largest=False)
                    for miner in eligible_miners[:available_slots]:
                        if self.composite_scores[miner] > lowest_next_tier_scores[-1]:
                            # Promote miner
                            new_tiers[miner] = next_tier
                            # Demote the lowest miner in the next tier
                            demote_miner = (new_tiers == next_tier).nonzero(as_tuple=True)[0][lowest_indices[-1]]
                            new_tiers[demote_miner] = tier
                            self.logger.info(f"Miner {miner} promoted to tier {next_tier}, Miner {demote_miner} demoted to tier {tier}")

                else:
                    # Promote miners up to available slots
                    promote_miners = eligible_miners[:available_slots]
                    new_tiers[promote_miners] = next_tier
                    for miner in promote_miners:
                        self.logger.info(f"Miner {miner} promoted to tier {next_tier}")

            self.tiers = new_tiers

            self.logger.info("Tier management completed")
            self.log_tier_summary()

        except Exception as e:
            self.logger.error(f"Error managing tiers: {str(e)}")
            raise

    def log_tier_summary(self):
        """
        Log the summary of tier distributions.
        """
        for tier in range(1, len(self.tier_configs) + 1):
            tier_count = (self.tiers == tier).sum().item()
            self.logger.info(f"Tier {tier}: {tier_count} miners")

    def calculate_weights(self) -> t.Tensor:
        """
        Calculate weights for all miners based on their tier and composite score.
        Weights sum to 1 and represent both the miner's share of incentives and their influence.
        """
        self.logger.info("Calculating weights")
        
        try:
            composite_scores = self.calculate_composite_scores()
            weights = t.zeros_like(composite_scores)
            
            tier_incentives = t.tensor([config['incentive'] for config in self.tier_configs])
            total_incentive = tier_incentives.sum()
            normalized_incentives = tier_incentives / total_incentive
            
            # Calculate the number of miners in each tier
            tier_counts = t.bincount(self.tiers, minlength=len(self.tier_configs)+1)[1:]
            
            # Only consider non-empty tiers
            non_empty_tiers = tier_counts > 0
            active_tiers = t.arange(1, len(self.tier_configs)+1)[non_empty_tiers]
            
            # Redistribute weights from empty tiers
            total_active_incentive = normalized_incentives[non_empty_tiers].sum()
            adjusted_incentives = normalized_incentives[non_empty_tiers] / total_active_incentive
            
            for i, tier in enumerate(active_tiers):
                tier_mask = self.tiers == tier
                tier_scores = composite_scores[tier_mask]
                
                if tier_scores.numel() == 0:
                    continue
                
                # Normalize scores within the tier
                tier_weights = tier_scores / tier_scores.sum()
                
                # Apply adjusted tier incentive
                tier_weights *= adjusted_incentives[i]
                
                weights[tier_mask] = tier_weights
            
            # Ensure weights sum to 1
            weights /= weights.sum()
            
            self.logger.info(f"Weights calculated - min: {weights.min().item():.4f}, max: {weights.max().item():.4f}, mean: {weights.mean().item():.4f}")
        
        except Exception as e:
            self.logger.error(f"Error calculating weights: {str(e)}")
            raise
        
        return weights

    def log_score_summary(self):
        self.logger.info("Logging score summary...")
        for score_name, score_tensor in [
            ("CLV", self.clv_scores),
            ("ROI", self.roi_scores),
            ("Sortino", self.sortino_scores),
            ("Entropy", self.entropy_scores),
            ("Composite", self.composite_scores)
        ]:
            self.logger.info(f"{score_name} Scores - min: {score_tensor.min().item():.4f}, max: {score_tensor.max().item():.4f}, mean: {score_tensor.mean().item():.4f}")

    def scoring_run(self, current_date):
        """
        Perform a complete scoring run for a given date.
        
        Args:
        current_date: datetime object representing the date to score.
        
        Returns:
        Tensor of weights for each miner, or None if no data found or error occurs
        """
        try:
            if isinstance(current_date, str):
                current_date = datetime.strptime(current_date, '%Y-%m-%d').date()
            elif isinstance(current_date, datetime):
                current_date = current_date.date()

            date_str = current_date.strftime('%Y-%m-%d')
            
            predictions, results, closing_line_odds = self.scoring_data.preprocess_for_scoring(date_str)
            print(f"Debug: Got {len(predictions)} predictions, {results.shape[0]} results, and {closing_line_odds.shape} closing line odds")
            print(f"Debug: First prediction shape: {predictions[0].shape}")
            print(f"Debug: Closing line odds shape: {closing_line_odds.shape}")
            
            self._validate_inputs(predictions, closing_line_odds, results)

            if not predictions:
                self.logger.warning(f"No predictions found for date {date_str}")
                return None

            # Update scores
            self.clv_scores = self._update_clv(predictions, closing_line_odds)
            self.roi_scores = self._update_roi(predictions, results)
            self.sortino_scores = self._update_sortino()
            try:
                self.entropy_scores = self.entropy_system.update_ebdr_scores(predictions, closing_line_odds, results)
            except Exception as e:
                self.logger.error(f"Error in entropy score calculation: {str(e)}")
                # Set default entropy scores
                self.entropy_scores = t.zeros((self.num_miners, self.max_days))
            self.composite_scores = self._update_composite_scores()

            # Update last_update_date
            self.last_update_date = current_date
            
            # Calculate and return weights
            weights = self.calculate_weights()
            print(f"Debug: Calculated weights shape: {weights.shape}")
            return weights
        
        except Exception as e:
            self.logger.error(f"Error in scoring run: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_miner_stats(self, weights: t.Tensor, predictions: List[t.Tensor], results: t.Tensor) -> Dict[int, Dict]:
        """
        Prepare miner stats for database update.
        
        Args:
        weights: Tensor of weights for each miner
        predictions: List of tensors containing miner predictions
        results: Tensor of game results
        
        Returns:
        Dictionary of miner stats
        """
        miner_stats = {}
        current_date = datetime.now(timezone.utc)
        
        # Check if it's a new day (00:00 UTC)
        is_new_day = current_date.hour == 0 and current_date.minute == 0
        
        for i, (weight, miner_preds) in enumerate(zip(weights, predictions)):
            if miner_preds.numel() == 0:
                continue  # Skip miners with no predictions

            # Calculate basic stats
            num_predictions = miner_preds.size(0)
            predicted_outcomes = miner_preds[:, 1].long()
            game_ids = miner_preds[:, 0].long()
            predicted_odds = miner_preds[:, 2]
            
            # Match predictions with results
            actual_outcomes = results[game_ids]
            correct_predictions = (predicted_outcomes == actual_outcomes).float()
            
            # Calculate wins and losses
            wins = correct_predictions.sum().item()
            losses = num_predictions - wins

            # Calculate earnings and profit
            earnings = t.where(correct_predictions == 1, predicted_odds - 1, t.tensor(-1.0)).sum().item()
            wager_amount = num_predictions  # Assuming 1 unit wager per prediction
            profit = earnings - wager_amount

            # Update cash
            if is_new_day:
                cash = 1000 - wager_amount  # Reset to 1000 and subtract wager
            else:
                # Fetch current cash from database and subtract wager
                current_cash = self.scoring_data.get_miner_cash(i)  # You need to implement this method
                cash = current_cash - wager_amount

            miner_stats[i] = {
                'status': 'active',
                'cash': cash,
                'current_incentive': weight.item(),
                'current_tier': self.tiers[i].item(),
                'current_scoring_window': self.max_days,
                'current_composite_score': self.composite_scores[i, -1].item(),
                'current_sharpe_ratio': 0,  # You may want to calculate this if needed
                'current_sortino_ratio': self.sortino_scores[i, -1].item(),
                'current_roi': self.roi_scores[i, -1].item(),
                'current_clv_avg': self.clv_scores[i, -1].item(),
                'last_prediction_date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'earnings': earnings,
                'wager_amount': wager_amount,
                'profit': profit,
                'predictions': num_predictions,
                'wins': wins,
                'losses': losses
            }

        return miner_stats

