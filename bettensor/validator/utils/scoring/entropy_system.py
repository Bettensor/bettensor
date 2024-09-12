
import math
import torch as t
import bettensor as bt


class EntropySystem:
    '''
    The Entropy System is a component of the composite score, which measures the diversity of miner predictions. 
    The goal of the entropy system is to discourage copy trading and incentive farming.
    '''

    def __init__(self, max_capacity, max_days, ebdr_weight=0.1, entropy_window=7):
        self.max_capacity = max_capacity
        self.max_days = max_days
        self.ebdr_weight = ebdr_weight
        self.entropy_window = entropy_window
        self.ebdr_scores = t.zeros(max_capacity, max_days)
        self.current_day = 0
        self.prediction_order = {}
        self.prediction_history = {}

    def calculate_bookmaker_probabilities(self, odds):
        probs = [1 / odd for odd in odds]
        probs = t.tensor(probs)
        return probs / probs.sum()

    def calculate_event_entropy(self, event):
        miner_probs = t.tensor([1 / pred['odds'] for pred in event['predictions'].values()]).float()
        miner_probs /= miner_probs.sum()
        miner_entropy = -t.sum(miner_probs * t.log2(miner_probs + 1e-10))

        bookmaker_probs = self.calculate_bookmaker_probabilities(event['current_odds'])
        bookmaker_entropy = -t.sum(bookmaker_probs * t.log2(bookmaker_probs + 1e-10))

        wager_sizes = t.tensor([pred['wager_size'] for pred in event['predictions'].values()])
        wager_entropy = -t.sum((wager_sizes / wager_sizes.sum()) * t.log2(wager_sizes / wager_sizes.sum() + 1e-10))
        return miner_entropy + wager_entropy, bookmaker_entropy

    def calculate_miner_entropy_contribution(self, miner_id, event):
        miner_probs = t.tensor([1 / pred['odds'] for pred in event['predictions'].values()]).float()
        total_predictions = miner_probs.sum().item()
        if total_predictions == 0:
            bt.logging.warning(f"No predictions for event: {event}")
            return 0.0
        miner_probs /= total_predictions
        total_entropy = -t.sum(miner_probs * t.log2(miner_probs + 1e-10))

        miner_probs_without = miner_probs.clone()
        miner_index = list(event['predictions'].keys()).index(miner_id)
        if miner_index < len(miner_probs_without):
            miner_probs_without[miner_index] = 0
        else:
            bt.logging.warning(f"Miner ID {miner_id} is out of bounds for event {event['id']}")
            return 0.0

        total_without = miner_probs_without.sum().item()
        if total_without == 0:
            return total_entropy.item()
        miner_probs_without /= total_without
        entropy_without = -t.sum(miner_probs_without * t.log2(miner_probs_without + 1e-10))

        contribution = total_entropy.item() - entropy_without.item()
        timing_weight = 1 / (self.prediction_order[event['id']].index(miner_id) + 1)
        contribution *= timing_weight
        if math.isnan(contribution):
            bt.logging.warning(f"NaN contribution calculated for miner {miner_id} in event {event}")
            return 0.0
        return contribution

    def calculate_uniqueness_score(self, miner_prediction, consensus_prediction):
        outcome_diff = abs(miner_prediction['outcome'] - consensus_prediction['outcome'])
        wager_diff = abs(miner_prediction['wager_size'] - consensus_prediction['wager_size']) / consensus_prediction['wager_size']
        return (outcome_diff + wager_diff) / 2

    def calculate_consensus_prediction(self, event):
        predictions = list(event['predictions'].values())
        if not predictions:
            return {
                'outcome': 0,
                'wager_size': 0
            }
        return {
            'outcome': sum(p['outcome'] for p in predictions) / len(predictions),
            'wager_size': sum(p['wager_size'] for p in predictions) / len(predictions)
        }

    def update_ebdr_scores(self, miners, events):
        if self.current_day >= self.max_days:
            bt.logging.warning(f"Reached maximum number of days ({self.max_days}). Skipping EBDR score update.")
            return

        for event in events:
            if not event['predictions']:
                bt.logging.warning(f"No predictions for event {event['id']}. Skipping EBDR score update for this event.")
                continue

            self.prediction_order[event['id']] = list(event['predictions'].keys())
            miner_entropy, bookmaker_entropy = self.calculate_event_entropy(event)
            consensus_prediction = self.calculate_consensus_prediction(event)
            
            for miner_uid in event['predictions'].keys():
                miner_contribution = self.calculate_miner_entropy_contribution(miner_uid, event)
                uniqueness_score = self.calculate_uniqueness_score(event['predictions'][miner_uid], consensus_prediction)
                historical_uniqueness = self.calculate_historical_uniqueness(miner_uid)
                contrarian_bonus = self.calculate_contrarian_bonus(event['predictions'][miner_uid], consensus_prediction, event['actual_outcome'])
                
                ebdr_score = miner_contribution / (bookmaker_entropy + 1e-10)
                ebdr_score *= (1 + uniqueness_score)
                ebdr_score *= (1 + historical_uniqueness)
                ebdr_score *= contrarian_bonus
                
                if miner_uid < self.max_capacity:
                    current_score = self.ebdr_scores[miner_uid, self.current_day]
                    new_score = current_score * (1 - self.ebdr_weight) + ebdr_score * self.ebdr_weight
                    self.ebdr_scores[miner_uid, self.current_day] = new_score
                else:
                    bt.logging.warning(f"Miner ID {miner_uid} is out of bounds for EBDR scores tensor")
                
                self.update_prediction_history(miner_uid, event['predictions'][miner_uid])

        if self.current_day >= self.entropy_window:
            window_start = max(0, self.current_day - self.entropy_window + 1)
            window_scores = self.ebdr_scores[:, window_start:self.current_day+1]
            self.ebdr_scores[:, self.current_day] = t.mean(window_scores, dim=1)

        self.current_day = min(self.current_day + 1, self.max_days - 1)

    def update_prediction_history(self, miner_id, prediction=None):
        if miner_id not in self.prediction_history:
            self.prediction_history[miner_id] = []
        if prediction is not None:
            self.prediction_history[miner_id].append(prediction)

    def calculate_historical_uniqueness(self, miner_id):
        if miner_id not in self.prediction_history or not self.prediction_history[miner_id]:
            return 0.0
        
        miner_history = self.prediction_history[miner_id]
        all_histories = list(self.prediction_history.values())
        if not all_histories:
            return 0.0
        
        uniqueness = sum(1 for h in all_histories if h != miner_history) / len(all_histories)
        return uniqueness

    def get_ebdr_scores(self):
        return self.ebdr_scores

    def calculate_contrarian_bonus(self, miner_prediction, consensus_prediction, actual_outcome):
        if miner_prediction['outcome'] != consensus_prediction['outcome'] and miner_prediction['outcome'] == actual_outcome:
            return 1.5
        return 1.0

    def get_uniqueness_scores(self):
        uniqueness_scores = []
        for miner_id in self.prediction_history:
            uniqueness_scores.append(self.calculate_historical_uniqueness(miner_id))
        return t.tensor(uniqueness_scores)

    def get_contrarian_bonuses(self):
        return t.ones(self.max_capacity)

    def get_historical_uniqueness(self):
        uniqueness_scores = []
        for miner_id in self.prediction_history:
            uniqueness_scores.append(self.calculate_historical_uniqueness(miner_id))
        return t.tensor(uniqueness_scores)