from dataclasses import dataclass
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from huggingface_hub import PyTorchModelHubMixin
from bettensor.miner.database.database_manager import DatabaseManager
import time

@dataclass
class MinerConfig:
    model_prediction: bool = False

class SoccerPredictor:
    def __init__(self, model_name, label_encoder_path=None, team_averages_path=None, id=0, db_manager=None, miner_stats_handler=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_HFmodel(model_name)
        if label_encoder_path is None:
            label_encoder_path = os.path.join(os.path.dirname(__file__), '..','models', 'label_encoder.pkl')
        self.le = self.load_label_encoder(label_encoder_path)
        self.scaler = StandardScaler()
        if team_averages_path is None:
            team_averages_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'team_averages_last_5_games_aug.csv')
        self.team_averages_path = team_averages_path
        self.db_manager = db_manager
        self.miner_stats_handler = miner_stats_handler
        self.made_daily_predictions = False

        #params
        self.id : int = id
        self.model_on : bool = False
        self.wager_distribution_steepness : int = 10
        self.fuzzy_match_percentage : int = 80
        self.minimum_wager_amount : float = 20.0
        self.maximum_wager_amount : float = 1000
        self.top_n_games : int = 10
        self.last_param_update = 0
        self.param_refresh_interval = 300  # 5 minutes in seconds
        self.get_model_params(self.db_manager)


    def get_model_params(self,db_manager: DatabaseManager):
        current_time = time.time()
        if current_time - self.last_param_update >= self.param_refresh_interval:
            row = db_manager.get_model_params(self.id)
            self.model_on = row['model_on']
            self.wager_distribution_steepness = row['wager_distribution_steepness']
            self.fuzzy_match_percentage = row['fuzzy_match_percentage']
            self.minimum_wager_amount = row['minimum_wager_amount']
            self.maximum_wager_amount = row['max_wager_amount']
            self.top_n_games = row['top_n_games']
            self.last_param_update = current_time

 
    def check_max_wager_vs_miner_cash(self, max_wager):
        '''
        Return the lesser of the max_wager and the miner's cash for model wager distribution.

        '''
        miner_cash = self.miner_stats_handler.get_miner_cash()
        return min(max_wager, miner_cash)

    def load_label_encoder(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def get_HFmodel(self, model_name):
        try:
            import warnings
            warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True")
            model = PodosTransformer.from_pretrained(f"Bettensor/{model_name}").to(self.device);
            return model
        except Exception as e:
            print(f"Error pulling huggingface model: {e}")
            return None

    def preprocess_data(self, home_teams, away_teams, odds):
        odds = np.array(odds) 
        df = pd.DataFrame({
            'HomeTeam': home_teams,
            'AwayTeam': away_teams,
            'B365H': odds[:, 0],
            'B365D': odds[:, 1],
            'B365A': odds[:, 2]
        })
        
        encoded_teams = set(self.le.classes_)
        df['home_encoded'] = df['HomeTeam'].apply(lambda x: self.le.transform([x])[0] if x in encoded_teams else None)
        df['away_encoded'] = df['AwayTeam'].apply(lambda x: self.le.transform([x])[0] if x in encoded_teams else None)
        df = df.dropna(subset=['home_encoded', 'away_encoded'])
        
        team_averages_df = pd.read_csv(self.team_averages_path)
        home_stats = ['Team', 'HS', 'HST', 'HC', 'HO', 'HY', 'HR', 'WinStreakHome', 'LossStreakHome', 'HomeTeamForm']
        away_stats = ['Team', 'AS', 'AST', 'AC', 'AO', 'AY', 'AR', 'WinStreakAway', 'LossStreakAway', 'AwayTeamForm']

        df = df.merge(team_averages_df[home_stats], left_on='HomeTeam', right_on='Team').drop(columns=['Team'])
        df = df.merge(team_averages_df[away_stats], left_on='AwayTeam', right_on='Team').drop(columns=['Team'])
        
        features = [
            'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HO', 'AO', 'HY', 'AY', 'HR', 'AR',
            'B365H', 'B365D', 'B365A', 'home_encoded', 'away_encoded', 'WinStreakHome', 
            'LossStreakHome', 'WinStreakAway', 'LossStreakAway', 'HomeTeamForm', 'AwayTeamForm'
        ]
        return df[features]

    def recommend_wager_distribution(self, confidence_scores):
        max_daily_wager = self.check_max_wager_vs_miner_cash(self.maximum_wager_amount)
        min_wager = self.minimum_wager_amount
        top_n = self.top_n_games

        confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
        top_indices = np.argsort(confidence_scores)[-top_n:]
        top_confidences = confidence_scores[top_indices]
        sigmoids = 1 / (1 + np.exp(-10 * (top_confidences - 0.5)))
        normalized_sigmoids = sigmoids / np.sum(sigmoids)
        
        wagers = normalized_sigmoids * max_daily_wager
        wagers = np.maximum(wagers, min_wager)
        wagers = np.round(wagers, 2)
        
        if np.sum(wagers) > max_daily_wager:
            excess = np.sum(wagers) - max_daily_wager
            while excess > 0.01:
                wagers[wagers > min_wager] -= 0.01
                wagers = np.round(wagers, 2)
                excess = np.sum(wagers) - max_daily_wager
        
        final_wagers = [0.0] * len(confidence_scores)
        for idx, wager in zip(top_indices, wagers):
            final_wagers[idx] = wager
        
        return final_wagers

    def predict_games(self, home_teams, away_teams, odds, max_daily_wager=None, min_wager=None, top_n=None):
        df = self.preprocess_data(home_teams, away_teams, odds)
        x = self.scaler.fit_transform(df)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probs = nn.Softmax(dim=1)(outputs.cpu())
            confidence_scores, pred_labels = torch.max(probs, dim=1)

        outcome_map = {0: "Home Win", 1: "Tie", 2: "Away Win"}
        pred_outcomes = [outcome_map[label.item()] for label in pred_labels]

        confidence_scores = confidence_scores.cpu().numpy()
        wagers = self.recommend_wager_distribution(confidence_scores)

        results = []
        for i in range(len(home_teams)):
            result = {
                'Home Team': home_teams[i],
                'Away Team': away_teams[i],
                'PredictedOutcome': pred_outcomes[i],
                'ConfidenceScore': np.round(confidence_scores[i].item(), 2),
                'recommendedWager' : wagers[i]
            }
            results.append(result)
        
        return results

class PodosTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1, temperature=1):
        super(PodosTransformer, self).__init__()
        self.temperature = temperature

        self.projection = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.projection(x)
        x = x.unsqueeze(1)  
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        x = self.fc(x)

        if self.temperature != 1.0:
            x = x / self.temperature

        return x