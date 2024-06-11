from typing import List
import math
import numpy as np


import numpy as np

def distribute_scores(scores):
    # Convert the scores dictionary to a list of tuples and extract the values
    items = list(scores.items())
    values = np.array([item[1] for item in items])
    
    # Normalize values to the range [0, 1]
    min_val = np.min(values)
    max_val = np.max(values)
    normalized_values = (values - min_val) / (max_val - min_val)
    
    # Apply a non-linear transformation to concentrate values between 0.25 and 0.75
    transformed_values = 0.5 * (1 - np.cos(np.pi * normalized_values))  # cosine transformation
    
    # Update the dictionary with redistributed scores
    redistributed_scores = {items[i][0]: transformed_values[i] for i in range(len(items))}
    
    return redistributed_scores

print(distribute_scores(example_total_predictions))

example_prediction_data = [
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "teamGameId": 100001,
    "minerId": "miner6",
    "predictionDate": "2024-05-01T13:45:30Z",
    "predictedOutcome": "Team A wins"
  },
  {
    "id": "423e4567-e89b-12d3-a456-426614174003",
    "teamGameId": 100002,
    "minerId": "miner6",
    "predictionDate": "2024-05-02T14:30:00Z",
    "predictedOutcome": "Team B wins"
  },
  {
    "id": "723e4567-e89b-12d3-a456-426614174006",
    "teamGameId": 100003,
    "minerId": "miner6",
    "predictionDate": "2024-05-03T16:00:45Z",
    "predictedOutcome": "Draw"
  }
]

# Can reference local DB for predictions outcome from the game_id before calling this function
def evaluate_predictions(predictions_json, correct_outcome, game_id):
    """
    Evaluates predictions and prints "correct" or "incorrect" based on the predicted outcome.
    
    Parameters:
    predictions_json : str
        JSON string containing a list of predictions.
    correct_outcome : str
        The correct outcome to check against the predicted outcomes.
    game_id : int
        The game id to filter and evaluate the predictions.
    """
    predictions = json.loads(predictions_json)
    
    for prediction in predictions:
        if prediction["teamGameId"] == game_id:
            predicted_outcome = prediction["predictedOutcome"]
            if predicted_outcome == correct_outcome:
                print("correct")
            else:
                print("incorrect")