import bittensor as bt

class OutcomeHandler:
    @staticmethod
    def convert_outcome(numeric_outcome, predicted_outcome):
        if numeric_outcome is None or numeric_outcome == "Unfinished":
            return "Unfinished"
        
        outcomes = {0: "Team A Win", 1: "Team B Win", 2: "Tie"}
        game_outcome_str = outcomes.get(numeric_outcome, numeric_outcome)
        
        if game_outcome_str == predicted_outcome:
            return "Wager Won"
        elif game_outcome_str in outcomes.values():
            return "Wager Lost"
        else:
            return "Unknown"

    @staticmethod
    def update_stats(db_manager, miner_uid, prediction):
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT miner_lifetime_wins, miner_lifetime_losses, miner_lifetime_earnings, miner_lifetime_predictions
                FROM miner_stats
                WHERE miner_uid = ?
            """, (miner_uid,))
            stats = cursor.fetchone()
            
            if stats is None:
                bt.logging.error(f"No stats found for miner_uid: {miner_uid}")
                return

            wins, losses, earnings, total_predictions = stats
            wager = prediction['wager']
            odds = prediction['teamAodds'] if prediction['predictedOutcome'] == "Team A Win" else prediction['teamBodds'] if prediction['predictedOutcome'] == "Team B Win" else prediction['tieOdds']

            if prediction['outcome'] == "Wager Won":
                wins += 1
                earnings += wager * (odds - 1)  # Subtract the original wager amount
            elif prediction['outcome'] == "Wager Lost":
                losses += 1
                earnings -= wager
            elif prediction['outcome'] == "Unfinished":
                # Don't update wins, losses, or earnings for unfinished games
                pass
            else:
                bt.logging.warning(f"Unknown prediction outcome: {prediction['outcome']}")

            total_predictions += 1

            cursor.execute("""
                UPDATE miner_stats
                SET miner_lifetime_wins = ?,
                    miner_lifetime_losses = ?,
                    miner_lifetime_earnings = ?,
                    miner_lifetime_predictions = ?,
                    miner_win_loss_ratio = ?
                WHERE miner_uid = ?
            """, (wins, losses, earnings, total_predictions, wins / (wins + losses) if (wins + losses) > 0 else 0, miner_uid))