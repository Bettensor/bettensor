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
from bettensor.validator.utils.scoring.scoring_data import get_closed_predictions_for_day, get_closed_games_for_day
from typing import List, Tuple
from datetime import datetime, timezone


class ScoringSystem:
    def __init__(self, num_miners=256, max_days=45):

        self.num_miners = num_miners # we'll just initialize this with the maximum slots for the subnet, with the assumption that we'll always have validators taking 10-20 slots. 
        self.max_days = max_days #maximum scoring window is 45 days. 
        
        # Fixed size tensors for active miners

        #Score Component Tensors up to max_days length, which is the maximum scoring window. These operate like a circular buffer, tracking the history over max_days. 
        self.clv_scores = t.zeros(num_miners, max_days)
        self.sortino_scores = t.zeros(num_miners, max_days)
        self.sharpe_scores = t.zeros(num_miners, max_days)
        self.returns = t.zeros(num_miners, max_days, 2) # 0 is amount bet, 1 is amount won 
        self.roi_scores = t.zeros(num_miners, max_days)
        
        self.entropy_scores = t.zeros(num_miners, max_days)
        self.composite_scores = t.zeros(num_miners,max_days)
        

        # Change num_predictions to amount_wagered
        self.amount_wagered = t.zeros(num_miners, max_days)

        self.tiers = t.ones(num_miners, dtype=t.int)  # All miners start in tier 1
        
        self.tier_history = t.ones(num_miners, max_days, dtype=t.int) # 45 day history of tiers. 


        







        # WARNING: DO NOT CHANGE THESE VALUES. THEY WILL IMPACT VTRUST SIGNIFICANTLY. 
        self.clv_weight = 0.25
        self.roi_weight = 0.25
        self.ssi_weight = 0.25
        self.entropy_weight = 0.25
        self.entropy_window = 30

        # Update tier configurations
        self.tier_configs = [
            {'window': 3, 'min_wager': 0, 'capacity': int(num_miners * 1.0), 'incentive': 0.1},
            {'window': 7, 'min_wager': 4000, 'capacity': int(num_miners * 0.2), 'incentive': 0.15},
            {'window': 15, 'min_wager': 10000, 'capacity': int(num_miners * 0.2), 'incentive': 0.2},
            {'window': 30, 'min_wager': 20000, 'capacity': int(num_miners * 0.1), 'incentive': 0.25},
            {'window': 45, 'min_wager': 35000, 'capacity': int(num_miners * 0.05), 'incentive': 0.3}
        ]

        # Change predictions storage to handle variable-length data
        self.daily_predictions = [[] for _ in range(max_days)]  # List of lists to store daily predictions

        self.last_update_date = None  # Store the date of the last update

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
        self.sharpe_scores = t.roll(self.sharpe_scores, shifts=-1, dims=1)
        self.returns = t.roll(self.returns, shifts=-1, dims=1)
        self.roi_scores = t.roll(self.roi_scores, shifts=-1, dims=1)
        self.entropy_scores = t.roll(self.entropy_scores, shifts=-1, dims=1)
        self.composite_scores = t.roll(self.composite_scores, shifts=-1, dims=1)
        self.amount_wagered = t.roll(self.amount_wagered, shifts=-1, dims=1)
        self.tier_history = t.roll(self.tier_history, shifts=-1, dims=1)

    def update_daily_scores(self, predictions: List[t.Tensor], closing_line_odds: t.Tensor, results: t.Tensor, debug: bool = True):
        """
        Updates the daily scores for all miners.
        
        Args:
        predictions: List of tensors, each of shape (num_predictions, 3) where the last dimension contains
                     [game_id, prediction, submitted_odds]
        closing_line_odds: Tensor of shape (num_games, 2) where the last dimension contains
                           [game_id, closing_odds]
        results: Tensor of shape (num_games,) containing the actual results (0 or 1)
        debug: If True, perform input validation checks (default: True)
        """
        current_date = self._get_current_date()
        
        if debug:
            self._validate_inputs(predictions, closing_line_odds, results)

        self.logger.info(f"Updating daily scores for date {current_date}")
        
        try:
            if not predictions or closing_line_odds.numel() == 0 or results.numel() == 0:
                self.logger.warning(f"No valid data for scoring on date {current_date}")
                return

            if self._should_shift(current_date):
                self._shift_tensors()
                self.last_update_date = current_date
            
            self.daily_predictions[-1] = predictions
            
            new_clv_scores = self._update_clv(predictions, closing_line_odds)
            new_roi_scores = self._update_roi(predictions, results)
            new_sharpe_scores = self._update_sharpe()
            new_sortino_scores = self._update_sortino()
            new_amount_wagered = self._calculate_amount_wagered(predictions)
            
            # Update the last position of each tensor
            self.clv_scores[:, -1] = new_clv_scores
            self.roi_scores[:, -1] = new_roi_scores
            self.sharpe_scores[:, -1] = new_sharpe_scores
            self.sortino_scores[:, -1] = new_sortino_scores
            self.amount_wagered[:, -1] = new_amount_wagered
            
            # Update composite scores
            new_composite_scores = self._update_composite_scores()
            self.composite_scores[:, -1] = new_composite_scores

            # Update tier history
            self.tier_history[:, -1] = self.tiers

        except Exception as e:
            self.logger.error(f"Error updating scores for date {current_date}: {str(e)}")
            raise
        
        self.logger.info(f"Daily scores updated for date {current_date}")
        self.log_score_summary()

    def _validate_inputs(self, predictions: List[t.Tensor], closing_line_odds: t.Tensor, results: t.Tensor):
        if not isinstance(predictions, list) or not all(isinstance(p, t.Tensor) and p.dim() == 2 and p.size(1) == 3 for p in predictions):
            raise ValueError("predictions must be a list of 2D tensors with shape (num_predictions, 3)")
        if not isinstance(closing_line_odds, t.Tensor) or closing_line_odds.dim() != 2 or closing_line_odds.size(1) != 2:
            raise ValueError("closing_line_odds must be a 2D tensor with shape (num_games, 2)")
        if not isinstance(results, t.Tensor) or results.dim() != 1:
            raise ValueError("results must be a 1D tensor")

    def _update_clv(self, predictions: List[t.Tensor], closing_line_odds: t.Tensor) -> t.Tensor:
        clv_scores = t.zeros(self.num_miners)
        for i, miner_predictions in enumerate(predictions):
            if miner_predictions.size(0) > 0:
                submitted_odds = miner_predictions[:, 2]
                game_ids = miner_predictions[:, 0].long()
                closing_odds = closing_line_odds[game_ids, 1]
                clv = (1 / submitted_odds - 1 / closing_odds) * 100
                clv_scores[i] = clv.mean()
        return clv_scores

    def _update_roi(self, predictions: List[t.Tensor], results: t.Tensor) -> t.Tensor:
        roi_scores = t.zeros(self.num_miners)
        for i, miner_predictions in enumerate(predictions):
            if miner_predictions.size(0) > 0:
                game_ids = miner_predictions[:, 0].long()
                miner_predictions = miner_predictions[:, 1]
                submitted_odds = miner_predictions[:, 2]
                prediction_results = results[game_ids]
                returns = t.where(prediction_results == miner_predictions, submitted_odds - 1, -1)
                roi_scores[i] = returns.mean()
        return roi_scores

    def _update_sharpe(self) -> t.Tensor:
        returns = self.roi_scores
        return (returns.mean(dim=1) / returns.std(dim=1)).nan_to_num(0)

    def _update_sortino(self) -> t.Tensor:
        returns = self.roi_scores
        negative_returns = t.where(returns < 0, returns, t.zeros_like(returns))
        downside_deviation = t.sqrt((negative_returns ** 2).mean(dim=1))
        return (returns.mean(dim=1) / downside_deviation).nan_to_num(0)

    def _calculate_amount_wagered(self, predictions: List[t.Tensor]) -> t.Tensor:
        return t.tensor([p.size(0) for p in predictions]).float()

    def _update_composite_scores(self) -> t.Tensor:
        return (
            self.clv_scores.mean(dim=1) * self.clv_weight +
            self.roi_scores.mean(dim=1) * self.roi_weight +
            self.sharpe_scores.mean(dim=1) * self.ssi_weight +
            self.sortino_scores.mean(dim=1) * self.ssi_weight +
            self.entropy_scores.mean(dim=1) * self.entropy_weight
        )

    def calculate_composite_scores(self):
        """
        Calculate composite scores for all miners based on their current tier.
        """
        composite_scores = t.zeros(self.num_miners)
        for tier, config in enumerate(self.tier_configs, 1):
            tier_mask = self.tiers == tier
            window = config['window']
            
            # Calculate composite score for miners in this tier
            tier_scores = (
                self.clv_scores[tier_mask, -window:].mean(dim=1) +
                self.roi_scores[tier_mask, -window:].mean(dim=1) +
                self.sharpe_scores[tier_mask, -window:].mean(dim=1) +
                self.sortino_scores[tier_mask, -window:].mean(dim=1)
            )
            
            composite_scores[tier_mask] = tier_scores
        
        return composite_scores

    def reset_miner(self, miner_uid: int):
        """
        Initialize a new miner, replacing any existing miner at the same UID. This should be called when the validator discovers a new miner in the metagraph.
        """
        # Reset scores for the miner
        self.clv_scores[miner_uid] = 0
        self.sortino_scores[miner_uid] = 0
        self.sharpe_scores[miner_uid] = 0
        self.roi_scores[miner_uid] = 0
        self.amount_wagered[miner_uid] = 0
        self.composite_scores[miner_uid] = 0
        self.entropy_scores[miner_uid] = 0
        
        # Set miner to tier 1
        self.tiers[miner_uid] = 1
        self.tier_history[miner_uid] = 1

    def calculate_composite_score(self, miner_uid: int, window: int) -> float:
        """
        Calculate the composite score for a miner over a given window.
        
        Args:
        miner_uid: The unique identifier of the miner
        window: The number of days to consider for the score calculation
        
        Returns:
        The composite score for the miner
        """
        scores = (
            self.clv_scores[miner_uid, -window:].mean() +
            self.roi_scores[miner_uid, -window:].mean() +
            self.sharpe_scores[miner_uid, -window:].mean() +
            self.sortino_scores[miner_uid, -window:].mean() +
            self.entropy_scores[miner_uid, -window:].mean() 
        )
        return scores.item()

    def check_demotion_eligibility(self, debug: bool = True):
        """
        Check demotion eligibility for all miners simultaneously using tensor operations.
        
        Args:
        debug: If True, perform input validation checks (default: True)
        
        Returns:
        Tensor of shape (num_miners,) containing the new tier for each miner
        """
        self.logger.info("Checking demotion eligibility")
        
        # Store current tiers and initialize new tiers
        current_tiers = self.tiers
        new_tiers = current_tiers.clone()

        # Calculate composite scores for all miners
        composite_scores = self.calculate_composite_scores()

        # Calculate wager amounts for all miners and tiers at once
        wager_amounts = t.stack([
            self.amount_wagered[:, -config['window']:].sum(dim=1)
            for config in self.tier_configs
        ], dim=1)

        # Create a tensor of minimum wager requirements for each tier
        min_wagers = t.tensor([config['min_wager'] for config in self.tier_configs])

        # Check minimum wager requirements for all tiers simultaneously
        demotion_mask = wager_amounts < min_wagers.unsqueeze(0)

        # Identify miners eligible for demotion
        demotion_candidates = t.where((current_tiers > 1) & demotion_mask[current_tiers - 1])[0]

        # Process each demotion candidate
        if len(demotion_candidates) > 0:
            for miner in demotion_candidates:
                current_tier = current_tiers[miner].item()
                # Check each lower tier, starting from the immediate lower tier
                for new_tier in range(current_tier - 1, 0, -1):
                    lower_tier_mask = current_tiers == new_tier
                    lower_tier_count = lower_tier_mask.sum()
                    lower_tier_capacity = self.tier_configs[new_tier - 1]['capacity']

                    if lower_tier_count < lower_tier_capacity:
                        # If there's capacity in the lower tier, demote the miner
                        new_tiers[miner] = new_tier
                        break
                    else:
                        # If the lower tier is at capacity, compare scores
                        lower_window = self.tier_configs[new_tier - 1]['window']
                        miner_score = self.calculate_composite_score(miner, lower_window)
                        lower_tier_scores = composite_scores[lower_tier_mask]
                        min_score, min_score_index = lower_tier_scores.min(dim=0)

                        if miner_score > min_score:
                            # If the miner's score is better, demote them and cascade the demotion
                            new_tiers[miner] = new_tier
                            cascade_miner = t.where(lower_tier_mask)[0][min_score_index]
                            new_tiers[cascade_miner] = new_tier - 1
                            break
                else:
                    # If no suitable tier was found, demote to the lowest tier
                    new_tiers[miner] = 1

        # Perform debug checks if enabled
        if debug:
            if not t.all((new_tiers >= 1) & (new_tiers <= len(self.tier_configs))):
                raise ValueError("Invalid tier assignments after demotion check")

        return new_tiers

    def check_promotion_eligibility(self):
        """
        Check promotion eligibility for all miners simultaneously.
        """
        current_tiers = self.tiers
        next_tiers = current_tiers.clone()
        
        for tier, config in enumerate(self.tier_configs[:-1], 1):  # Exclude the highest tier
            tier_mask = current_tiers == tier
            next_tier = tier + 1
            window = self.tier_configs[next_tier - 1]['window']
            min_wager = self.tier_configs[next_tier - 1]['min_wager']
            
            # Check minimum wager requirement
            wager_amounts = self.amount_wagered[:, -window:].sum(dim=1)
            eligible_mask = (wager_amounts >= min_wager) & tier_mask
            
            # Check if next tier is at capacity
            next_tier_count = (current_tiers == next_tier).sum()
            next_tier_capacity = self.tier_configs[next_tier - 1]['capacity']
            
            if next_tier_count < next_tier_capacity:
                # Promote all eligible miners
                next_tiers[eligible_mask] = next_tier
            else:
                # Compare scores with lowest scoring miner in next tier
                composite_scores = self.calculate_composite_scores()
                next_tier_mask = current_tiers == next_tier
                next_tier_min_score = composite_scores[next_tier_mask].min()
                
                promotion_mask = (composite_scores > next_tier_min_score) & eligible_mask
                demotion_mask = (composite_scores == next_tier_min_score) & next_tier_mask
                
                next_tiers[promotion_mask] = next_tier
                next_tiers[demotion_mask] = tier
        
        return next_tiers

    def manage_tiers(self, debug: bool = True):
        """
        Manage tier demotions and promotions for all miners.
        
        Args:
        debug: If True, perform input validation checks (default: True)
        """
        self.logger.info("Managing tiers")
        
        try:
            old_tiers = self.tiers.clone()
            
            # First, handle demotions
            new_tiers = self.check_demotion_eligibility(debug=debug)
            self.tiers = new_tiers
            
            # Then, handle promotions
            new_tiers = self.check_promotion_eligibility()
            
            if debug:
                if not t.all((new_tiers >= 1) & (new_tiers <= len(self.tier_configs))):
                    raise ValueError("Invalid tier assignments")
            
            self.tiers = new_tiers
            
            demotions = (new_tiers < old_tiers).sum().item()
            promotions = (new_tiers > old_tiers).sum().item()
            self.logger.info(f"Tier changes: {demotions} demotions, {promotions} promotions")
            
            for tier in range(1, len(self.tier_configs) + 1):
                tier_count = (self.tiers == tier).sum().item()
                self.logger.info(f"Tier {tier}: {tier_count} miners")
                
                if debug:
                    max_capacity = self.tier_configs[tier-1]['capacity']
                    if tier_count > max_capacity:
                        self.logger.warning(f"Tier {tier} exceeds maximum capacity: {tier_count} > {max_capacity}")
        
        except Exception as e:
            self.logger.error(f"Error managing tiers: {str(e)}")
            raise

    def calculate_final_weights(self, debug: bool = True) -> t.Tensor:
        """
        Calculate the final weights for all miners based on their tier and composite score.
        
        Args:
        debug: If True, perform input validation checks (default: True)
        
        Returns:
        Tensor of shape (num_miners,) containing the final weights for each miner
        """
        self.logger.info("Calculating final weights")
        
        try:
            composite_scores = self.calculate_composite_scores()
            weights = t.zeros_like(composite_scores)
            
            for tier, config in enumerate(self.tier_configs, 1):
                tier_mask = self.tiers == tier
                tier_scores = composite_scores[tier_mask]
                
                if debug and tier_scores.numel() == 0:
                    self.logger.warning(f"No miners in tier {tier}")
                    continue
                
                # Normalize scores within the tier
                tier_weights = tier_scores / tier_scores.sum()
                
                # Apply tier incentive
                tier_weights *= config['incentive']
                
                weights[tier_mask] = tier_weights
            
            # Normalize weights to sum to 1
            total_weight = weights.sum()
            if debug and total_weight == 0:
                raise ValueError("Total weight is zero, cannot normalize")
            weights /= total_weight
            
        except Exception as e:
            self.logger.error(f"Error calculating final weights: {str(e)}")
            raise
        
        self.log_weight_summary(weights)
        
        return weights


    def scoring_run(self):
        '''
        Perform a full scoring run, called from the main validator loop before setting weights.
        '''

        #TODO: preprocess data (scoring_data.py)
        # Get tensors of miner predictions, closing line odds, and results for a given day. 



        #TODO: calculate component score updates with new predictions
        # CLV, ROI, SSI, Entropy over the appropriate windows.

        #TODO: calculate composite score updates 

        #TODO: manage tier changes 

        #TODO: calculate final weights

        #TODO: return final weights for weight setting.

        pass







        