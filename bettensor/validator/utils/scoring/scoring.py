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


class ScoringSystem:
    def __init__(self, num_miners=256, max_days=45):
        self.num_miners = num_miners
        self.max_days = max_days
        
        # Centralized data storage
        self.clv_scores = t.zeros(num_miners, max_days)
        self.sortino_scores = t.zeros(num_miners, max_days)
        self.sharpe_scores = t.zeros(num_miners, max_days)
        self.roi_scores = t.zeros(num_miners, max_days)
        self.num_predictions = t.zeros(num_miners, max_days, dtype=t.int)
        self.tiers = t.ones(num_miners, dtype=t.int)  # All miners start in tier 1


        # TODO: Composite score weights - should be set by a config file maybe? 


        # WARNING: DO NOT CHANGE THESE VALUES. THEY WILL IMPACT VTRUST SIGNIFICANTLY. 
        self.clv_weight = 0.25
        self.roi_weight = 0.25
        self.ssi_weight = 0.25


        
        # Tier configurations
        self.tier_configs = [
            {'window': 3, 'min_predictions': 0, 'capacity': int(num_miners * 1.0), 'incentive': 0.1},
            {'window': 7, 'min_predictions': 10, 'capacity': int(num_miners * 0.2), 'incentive': 0.15},
            {'window': 15, 'min_predictions': 30, 'capacity': int(num_miners * 0.2), 'incentive': 0.2},
            {'window': 30, 'min_predictions': 65, 'capacity': int(num_miners * 0.1), 'incentive': 0.25},
            {'window': 45, 'min_predictions': 100, 'capacity': int(num_miners * 0.05), 'incentive': 0.3}
        ]

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def update_daily_scores(self, day: int, predictions: t.Tensor, closing_line_odds: t.Tensor, results: t.Tensor, debug: bool = True):
        """
        Updates the daily scores for all miners.
        
        Args:
        day: The current day index
        predictions: Tensor of shape (num_miners, num_predictions, 3) where the last dimension contains
                     [game_id, prediction, submitted_odds]
        closing_line_odds: Tensor of shape (num_games, 2) where the last dimension contains
                           [game_id, closing_odds]
        results: Tensor of shape (num_games,) containing the actual results (0 or 1)
        debug: If True, perform input validation checks (default: True)
        """
        if debug:
            if not isinstance(day, int) or day < 0 or day >= self.max_days:
                raise ValueError(f"day must be an integer between 0 and {self.max_days-1}")
            if not isinstance(predictions, t.Tensor) or predictions.dim() != 3 or predictions.size(2) != 3:
                raise ValueError("predictions must be a 3D tensor with shape (num_miners, num_predictions, 3)")
            if not isinstance(closing_line_odds, t.Tensor) or closing_line_odds.dim() != 2 or closing_line_odds.size(1) != 2:
                raise ValueError("closing_line_odds must be a 2D tensor with shape (num_games, 2)")
            if not isinstance(results, t.Tensor) or results.dim() != 1:
                raise ValueError("results must be a 1D tensor")

        self.logger.info(f"Updating daily scores for day {day}")
        
        try:
            self.clv_scores[:, day] = self.calculate_clv(predictions, closing_line_odds, debug=debug)
            self.roi_scores[:, day] = self.calculate_roi(predictions, results, debug=debug)
            self.sharpe_scores[:, day] = self.calculate_sharpe(self.roi_scores[:, :day+1], debug=debug)
            self.sortino_scores[:, day] = self.calculate_sortino(self.roi_scores[:, :day+1], debug=debug)
            self.num_predictions[:, day] = self.count_predictions(predictions, debug=debug)
        except Exception as e:
            self.logger.error(f"Error updating scores for day {day}: {str(e)}")
            raise
        
        self.logger.info(f"Daily scores updated for day {day}")
        self.log_score_summary(day)

    def log_score_summary(self, day: int):
        """Log a summary of the scores for the current day."""
        self.logger.info(f"Score summary for day {day}:")
        self.logger.info(f"CLV scores - Mean: {self.clv_scores[:, day].mean():.4f}, Max: {self.clv_scores[:, day].max():.4f}")
        self.logger.info(f"ROI scores - Mean: {self.roi_scores[:, day].mean():.4f}, Max: {self.roi_scores[:, day].max():.4f}")
        self.logger.info(f"Sharpe scores - Mean: {self.sharpe_scores[:, day].mean():.4f}, Max: {self.sharpe_scores[:, day].max():.4f}")
        self.logger.info(f"Sortino scores - Mean: {self.sortino_scores[:, day].mean():.4f}, Max: {self.sortino_scores[:, day].max():.4f}")
        self.logger.info(f"Total predictions: {self.num_predictions[:, day].sum()}")

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

    def calculate_clv(self, predictions: t.Tensor, closing_line_odds: t.Tensor, debug: bool = True) -> t.Tensor:
        """
        Calculate Closing Line Value (CLV) for all miners.
        
        Args:
        predictions: Tensor of shape (num_miners, num_predictions, 3) where the last dimension contains
                     [game_id, prediction, submitted_odds]
        closing_line_odds: Tensor of shape (num_games, 2) where the last dimension contains
                           [game_id, closing_odds]
        debug: If True, perform input validation checks (default: True)
        
        Returns:
        Tensor of shape (num_miners,) containing the average CLV for each miner
        """
        if debug:
            if not isinstance(predictions, t.Tensor) or not isinstance(closing_line_odds, t.Tensor):
                raise TypeError("Both predictions and closing_line_odds must be PyTorch tensors")
            
            if predictions.dim() != 3 or predictions.size(2) != 3:
                raise ValueError("predictions must have shape (num_miners, num_predictions, 3)")
            
            if closing_line_odds.dim() != 2 or closing_line_odds.size(1) != 2:
                raise ValueError("closing_line_odds must have shape (num_games, 2)")
            
            if predictions[:, :, 0].long().max() >= closing_line_odds.size(0):
                raise ValueError("Game ID in predictions is out of range for closing_line_odds")
        
        submitted_odds = predictions[:, :, 2]
        game_ids = predictions[:, :, 0].long()
        
        closing_odds = closing_line_odds[game_ids]
        clv = (1 / submitted_odds - 1 / closing_odds) * 100
        
        return clv.mean(dim=1)

    def calculate_roi(self, predictions: t.Tensor, results: t.Tensor, debug: bool = True) -> t.Tensor:
        """
        Calculate Return on Investment (ROI) for all miners.
        
        Args:
        predictions: Tensor of shape (num_miners, num_predictions, 3) where the last dimension contains
                     [game_id, prediction, submitted_odds]
        results: Tensor of shape (num_games,) containing the actual results (0 or 1)
        debug: If True, perform input validation checks (default: True)
        
        Returns:
        Tensor of shape (num_miners,) containing the ROI for each miner
        """
        if debug:
            if not isinstance(predictions, t.Tensor) or not isinstance(results, t.Tensor):
                raise TypeError("Both predictions and results must be PyTorch tensors")
            
            if predictions.dim() != 3 or predictions.size(2) != 3:
                raise ValueError("predictions must have shape (num_miners, num_predictions, 3)")
            
            if results.dim() != 1:
                raise ValueError("results must be a 1-dimensional tensor")
            
            if predictions[:, :, 0].long().max() >= results.size(0):
                raise ValueError("Game ID in predictions is out of range for results")
        
        game_ids = predictions[:, :, 0].long()
        miner_predictions = predictions[:, :, 1]
        submitted_odds = predictions[:, :, 2]
        
        prediction_results = results[game_ids]
        returns = t.where(prediction_results == miner_predictions, submitted_odds - 1, -1)
        
        total_returns = returns.sum(dim=1)
        num_bets = (submitted_odds > 0).sum(dim=1)
        
        roi = t.where(num_bets > 0, total_returns / num_bets, t.zeros_like(total_returns))
        
        return roi

    def calculate_sharpe(self, daily_returns: t.Tensor, risk_free_rate: float = 0.0, debug: bool = True) -> t.Tensor:
        """
        Calculate Sharpe Ratio for all miners.
        
        Args:
        daily_returns: Tensor of shape (num_miners, num_days) containing daily returns
        risk_free_rate: The risk-free rate of return (default: 0.0)
        debug: If True, perform input validation checks (default: True)
        
        Returns:
        Tensor of shape (num_miners,) containing the Sharpe Ratio for each miner
        """
        if debug:
            if not isinstance(daily_returns, t.Tensor):
                raise TypeError("daily_returns must be a PyTorch tensor")
            
            if daily_returns.dim() != 2:
                raise ValueError("daily_returns must have shape (num_miners, num_days)")
            
            if not isinstance(risk_free_rate, (int, float)):
                raise TypeError("risk_free_rate must be a number")
        
        excess_returns = daily_returns - risk_free_rate
        mean_excess_returns = excess_returns.mean(dim=1)
        std_excess_returns = excess_returns.std(dim=1)
        
        sharpe_ratio = t.where(std_excess_returns != 0, 
                               mean_excess_returns / std_excess_returns,
                               t.zeros_like(mean_excess_returns))
        
        return sharpe_ratio

    def calculate_sortino(self, daily_returns: t.Tensor, risk_free_rate: float = 0.0, debug: bool = True) -> t.Tensor:
        """
        Calculate Sortino Ratio for all miners.
        
        Args:
        daily_returns: Tensor of shape (num_miners, num_days) containing daily returns
        risk_free_rate: The risk-free rate of return (default: 0.0)
        debug: If True, perform input validation checks (default: True)
        
        Returns:
        Tensor of shape (num_miners,) containing the Sortino Ratio for each miner
        """
        if debug:
            if not isinstance(daily_returns, t.Tensor):
                raise TypeError("daily_returns must be a PyTorch tensor")
            
            if daily_returns.dim() != 2:
                raise ValueError("daily_returns must have shape (num_miners, num_days)")
            
            if not isinstance(risk_free_rate, (int, float)):
                raise TypeError("risk_free_rate must be a number")
        
        excess_returns = daily_returns - risk_free_rate
        mean_excess_returns = excess_returns.mean(dim=1)
        
        negative_returns = t.where(excess_returns < 0, excess_returns, t.zeros_like(excess_returns))
        downside_deviation = t.sqrt((negative_returns ** 2).mean(dim=1))
        
        sortino_ratio = t.where(downside_deviation != 0, 
                                mean_excess_returns / downside_deviation,
                                t.zeros_like(mean_excess_returns))
        
        return sortino_ratio

    def count_predictions(self, predictions: t.Tensor, debug: bool = True) -> t.Tensor:
        """
        Count the number of predictions for each miner.
        
        Args:
        predictions: Tensor of shape (num_miners, num_predictions, 3)
        debug: If True, perform input validation checks (default: True)
        
        Returns:
        Tensor of shape (num_miners,) containing the count of predictions for each miner
        """
        if debug:
            if not isinstance(predictions, t.Tensor):
                raise TypeError("predictions must be a PyTorch tensor")
            
            if predictions.dim() != 3 or predictions.size(2) != 3:
                raise ValueError("predictions must have shape (num_miners, num_predictions, 3)")
        
        return (predictions[:, :, 2] > 0).sum(dim=1)

    def register_miner(self, miner_uid: int):
        """
        Register a new miner, replacing any existing miner at the same UID.
        """
        # Reset scores for the miner
        self.clv_scores[miner_uid] = 0
        self.sortino_scores[miner_uid] = 0
        self.sharpe_scores[miner_uid] = 0
        self.roi_scores[miner_uid] = 0
        self.num_predictions[miner_uid] = 0
        
        # Set miner to tier 1
        self.tiers[miner_uid] = 1

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
            self.sortino_scores[miner_uid, -window:].mean()
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

        # Calculate prediction counts for all miners and tiers at once
        # This creates a tensor of shape (num_miners, num_tiers)
        prediction_counts = t.stack([
            self.num_predictions[:, -config['window']:].sum(dim=1)
            for config in self.tier_configs
        ], dim=1)

        # Create a tensor of minimum prediction requirements for each tier
        min_predictions = t.tensor([config['min_predictions'] for config in self.tier_configs])

        # Check minimum prediction requirements for all tiers simultaneously
        # This creates a boolean mask of shape (num_miners, num_tiers)
        # where True indicates the miner doesn't meet the minimum predictions for that tier
        demotion_mask = prediction_counts < min_predictions.unsqueeze(0)

        # Identify miners eligible for demotion
        # These are miners who are not in the lowest tier (tier > 1)
        # and don't meet the minimum predictions for their current tier
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
            min_predictions = self.tier_configs[next_tier - 1]['min_predictions']
            
            # Check minimum prediction requirement
            prediction_counts = self.num_predictions[:, -window:].sum(dim=1)
            eligible_mask = (prediction_counts >= min_predictions) & tier_mask
            
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

    def log_weight_summary(self, weights: t.Tensor):
        """Log a summary of the final weights."""
        self.logger.info("Final weight summary:")
        for tier, config in enumerate(self.tier_configs, 1):
            tier_mask = self.tiers == tier
            tier_weights = weights[tier_mask]
            self.logger.info(f"Tier {tier}:")
            self.logger.info(f"  Miners: {tier_mask.sum()}")
            self.logger.info(f"  Total weight: {tier_weights.sum():.4f}")
            self.logger.info(f"  Mean weight: {tier_weights.mean():.4f}")
            self.logger.info(f"  Max weight: {tier_weights.max():.4f}")

    # def run_epoch(self, predictions: t.Tensor, closing_line_odds: t.Tensor, results: t.Tensor, debug: bool = True):
    #     """
    #     Run a full epoch of scoring and tier management.
        
    #     Args:
    #     predictions: Tensor of shape (num_miners, num_days, num_predictions, 3)
    #     closing_line_odds: Tensor of shape (num_days, num_games, 2)
    #     results: Tensor of shape (num_days, num_games)
    #     debug: If True, perform input validation checks (default: True)
        
    #     Returns:
    #     Tensor of shape (num_miners,) containing the final weights for each miner
    #     """
    #     self.logger.info("Starting new epoch")
        
    #     if debug:
    #         if not isinstance(predictions, t.Tensor) or predictions.dim() != 4 or predictions.size(3) != 3:
    #             raise ValueError("predictions must be a 4D tensor with shape (num_miners, num_days, num_predictions, 3)")
    #         if not isinstance(closing_line_odds, t.Tensor) or closing_line_odds.dim() != 3 or closing_line_odds.size(2) != 2:
    #             raise ValueError("closing_line_odds must be a 3D tensor with shape (num_days, num_games, 2)")
    #         if not isinstance(results, t.Tensor) or results.dim() != 2:
    #             raise ValueError("results must be a 2D tensor with shape (num_days, num_games)")
    #         if predictions.size(0) != self.num_miners or predictions.size(1) != self.max_days:
    #             raise ValueError(f"predictions must have shape ({self.num_miners}, {self.max_days}, num_predictions, 3)")
    #         if closing_line_odds.size(0) != self.max_days or results.size(0) != self.max_days:
    #             raise ValueError(f"closing_line_odds and results must have {self.max_days} days")
        
    #     try:
    #         for day in range(predictions.size(1)):
    #             self.update_daily_scores(day, predictions[:, day], closing_line_odds[day], results[day], debug=debug)
            
    #         self.manage_tiers(debug=debug)
    #         final_weights = self.calculate_final_weights(debug=debug)
        
    #     except Exception as e:
    #         self.logger.error(f"Error running epoch: {str(e)}")
    #         raise
        
    #     self.logger.info("Epoch completed")
    #     return final_weights
    def preprocess_data(self, date: str):
        '''
        Preprocesses the data for a given day.
        '''
        predictions = self.get_closed_predictions_for_day(date)
        games = self.get_closed_games_for_day(date)




        