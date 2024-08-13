from dataclasses import dataclass
import numpy as np
import pickle
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from huggingface_hub import PyTorchModelHubMixin

@dataclass
class MinerConfig:
    model_prediction: bool = False

class SoccerPredictor:
    def __init__(self, model_name, label_encoder_path='label_encoder.pkl'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_HFmodel(model_name)
        self.le = self.load_label_encoder(label_encoder_path)
        self.scaler = StandardScaler()
        self.team_averages_path = 'team_averages_last_5_games_aug.csv'

    def load_label_encoder(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def get_HFmodel(self, model_name):
        try:
            model = PodosTransformer.from_pretrained(f"Bettensor/{model_name}").to(self.device)
            return model
        except Exception as e:
            print(f"Error pulling huggingface model: {e}")
            return None

    def preprocess_data(self, home_teams, away_teams, odds):
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

    def predict_games(self, home_teams, away_teams, odds):
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

        results = []
        for i in range(len(home_teams)):
            result = {
                'Home Team': home_teams[i],
                'Away Team': away_teams[i],
                'PredictedOutcome': pred_outcomes[i],
                'ConfidenceScore': np.round(confidence_scores[i].item(), 2)
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