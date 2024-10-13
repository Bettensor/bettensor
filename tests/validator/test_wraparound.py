def test_wraparound(self):
    ...
    # Verify the loop variable after completion
    expected_day = (self.scoring_system.max_days + 5) % self.scoring_system.max_days
    self.assertEqual(
        self.scoring_system.current_day,
        expected_day,
        f"Current day {self.scoring_system.current_day} does not match expected day {expected_day}",
    )
