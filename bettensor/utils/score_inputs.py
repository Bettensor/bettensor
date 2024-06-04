

# Miners history of wins/losses must live on chain forever
# TODO: get miners inputs forever?
# TODO: 
# reward = (percentage_right(0.5) + difficulty_ranking(0.30) + total_predictions_last_2_days(0.20)) - 0.1*(hours_greater_than24_/6) so after 60 hours you literally cannot have any reward

from typing import List
import math
import numpy as np

def difficulty_ranking(all_odds: List[float]) -> float:
    """
    Returns a difficulty factor for predictions made by a Miner

    Parameters:
    all_odds : list of floats
        A list of float values representing odds that a miner has bet on.

    Returns
    float
        The difficulty factor of the miners bets.

    Raises
    TypeError
        If `all_odds` is not a list.
    ValueError
        If `all_odds` is an empty list.
    """
    # Verify type is a list
    if not isinstance(all_odds, list):
        raise TypeError("The input 'all_odds' must be a list")

    if len(all_odds) == 0:
        raise ValueError("all_odds must not be empty")

    average_odds = sum(all_odds) / len(all_odds)
    return float(average_odds)


#Not clustering properly in the center. Maybe a sigmoid isnt the right function?
example_total_predictions = {
    '1': 1.0,
    '2': 2.0,
    '3': 3.0,
    '4': 4.0,
    '5': 5.0,
    '6': 6.0,
    '7': 7.0,
    '8': 8.0,
    '9': 9.0,
    '10': 10.0,
    '11': 11.0,
    '12': 12.0,
    '13': 13.0,
    '14': 14.0,
    '15': 15.0,
    '16': 16.0,
    '17': 17.0,
    '18': 18.0,
    '19': 19.0,
    '20': 20.0,
    '21': 21.0,
    '22': 22.0,
    '23': 23.0,
    '24': 24.0,
    '25': 25.0,
    '26': 26.0,
    '27': 27.0,
    '28': 28.0,
    '29': 29.0,
    '30': 30.0,
    '31': 31.0,
    '32': 32.0,
    '33': 33.0,
    '34': 34.0,
    '35': 35.0,
    '36': 36.0,
    '37': 37.0,
    '38': 38.0,
    '39': 39.0,
    '40': 40.0
}



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