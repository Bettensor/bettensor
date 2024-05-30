

# Miners history of wins/losses must live on chain forever
# TODO: get miners inputs forever?
# TODO: 
# reward = (percentage_right(0.5) + difficulty_ranking(0.30) + total_predictions_last_2_days(0.20)) - 0.1*(hours_greater_than24_/6) so after 60 hours you literally cannot have any reward

from typing import List

def difficulty_ranking(all_odds: List[float]) -> float:
    """
    Returns a difficulty factor for predictions made by a Miner

    Parameters
    ----------
    all_odds : list of floats
        A list of float values representing odds that a miner has bet on.

    Returns
    -------
    float
        The difficulty factor of the miners bets.

    Raises
    ------
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


example_pred_data = [
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
    
    Parameters
    ----------
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