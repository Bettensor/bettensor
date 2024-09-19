def scoring_run(self, current_date):
    ...
    # Ensure days_since_reference is correctly calculated
    days_since_reference = (current_date - self.reference_date).days
    new_day = days_since_reference % self.max_days
    new_hour = current_date.hour

    if new_day != self.current_day or new_hour != self.current_hour:
        self.logger.info(
            f"Moving from day {self.current_day}, hour {self.current_hour} to day {new_day}, hour {new_hour}"
        )
        self._increment_time(new_day, new_hour)

    # Update current_day and current_hour correctly
    self.current_day = new_day
    self.current_hour = new_hour
    ...


def manage_tiers(self):
    self.logger.info("Managing tiers")

    try:
        composite_scores = self.calculate_composite_scores()
        current_tiers = self._get_tensor_for_day(
            self.tiers, self.current_day, self.current_hour
        )
        new_tiers = current_tiers.clone()

        initial_miner_count = (current_tiers > 0).sum()

        # Step 1: Handle demotions
        self._handle_demotions(new_tiers, composite_scores)

        # Step 2: Handle promotions, swaps, and fill empty slots
        self._handle_promotions_and_fill_slots(new_tiers, composite_scores)

        # Update tiers for the current day and hour
        self._set_tensor_for_day(
            self.tiers, self.current_day, self.current_hour, new_tiers
        )

        # Propagate the new tier information to the next day
        next_day = (self.current_day + 1) % self.max_days
        self.tiers[:, next_day, :] = new_tiers.unsqueeze(1).expand(
            -1, self.scores_per_day
        )

        final_miner_count = (new_tiers > 0).sum()
        assert (
            initial_miner_count == final_miner_count
        ), f"Miner count changed from {initial_miner_count} to {final_miner_count}"

        self.logger.info("Tier management completed")
        self.log_tier_summary()
    except Exception as e:
        self.logger.error(f"Error managing tiers: {str(e)}")
        raise


def _handle_promotions_and_fill_slots(self, tiers, composite_scores):
    for tier in range(1, len(self.tier_configs)):
        config = self.tier_configs[tier - 1]
        next_config = self.tier_configs[tier]

        # Handle promotions and swaps
        self._promote_and_swap(tier, tiers, composite_scores, config, next_config)

        # Fill empty slots
        self._fill_empty_slots(tier, tiers, composite_scores, config)

    # Ensure that higher tiers have better scores
    for tier_idx in range(len(self.tier_configs)):
        tier = tier_idx + 1
        tier_mask = tiers == tier
        # Corrected indexing to include current_day and current_hour
        tier_scores = composite_scores[
            tier_mask, self.current_day, self.current_hour, tier_idx
        ]
        if tier_scores.numel() > 0:
            sorted_scores, sorted_indices = tier_scores.sort(descending=True)
            top_miners = sorted_indices[
                : int(self.tier_configs[tier_idx]["capacity"] * 0.1)
            ]
            # Promote top miners to the next tier
            promote_mask = sorted_indices[
                : int(self.tier_configs[tier_idx]["capacity"] * 0.1)
            ]
            miner_ids = tier_mask.nonzero(as_tuple=True)[0][promote_mask]
            tiers[miner_ids] = tier + 1  # Promote to next tier


def _increment_time(self, new_day, new_hour):
    if new_day != self.current_day:
        target_day = self._get_day_index(new_day)
        source_day = self._get_day_index(self.current_day)
        self.tiers[:, target_day, :] = (
            self.tiers[:, source_day, self.current_hour]
            .unsqueeze(1)
            .expand(-1, self.scores_per_day)
        )

    self.current_day = new_day % self.max_days  # Ensure current_day wraps around
    self.current_hour = (
        new_hour % self.scores_per_day
    )  # Ensure current_hour wraps around
