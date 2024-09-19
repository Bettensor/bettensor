import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from bettensor.miner.database.predictions import PredictionsHandler
from bettensor.protocol import TeamGame, TeamGamePrediction
from bettensor.miner.stats.miner_stats import MinerStateManager
from bettensor.miner.models.model_utils import SoccerPredictor
import datetime
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockDBManager:
    def __init__(self):
        self.connection_pool = self

    def getconn(self):
        return self

    def putconn(self, conn):
        pass

    def cursor(self, cursor_factory=None):
        return self.get_cursor()

    def get_cursor(self):
        class MockCursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def execute(self, query, params=None):
                self.query = query
                self.params = params
                print(f"Executing query: {query} with params: {params}")

            def fetchall(self):
                return []

            def fetchone(self):
                if "FROM predictions WHERE minerID" in self.query:
                    if "SUM(wager)" in self.query:
                        return (100.0,)
                    return (10, 5, 5, 500.0, 1000.0, "2023-01-01")
                if "COALESCE(SUM(wager), 0)" in self.query:
                    return (100.0,)
                if "FROM miner_stats WHERE miner_hotkey" in self.query:
                    return None
                return None

            def close(self):
                pass

        return MockCursor()

    def execute_query(self, query, params=None):
        cursor = self.get_cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def commit(self):
        pass


@pytest.fixture
def mock_predictions_handler():
    db_manager = MockDBManager()
    miner_hotkey = "test_hotkey"
    miner_uid = "test_miner_uid"
    state_manager = MinerStateManager(db_manager, miner_hotkey, miner_uid)
    models = {"soccer": SoccerPredictor(model_name="podos_soccer_model")}
    return PredictionsHandler(db_manager, state_manager, miner_hotkey)


def fetch_mock_games():
    games = {
        "game1": TeamGame(
            id="game1",
            teamA="Manchester United",
            teamB="Liverpool",
            teamAodds=2.0,
            teamBodds=3.0,
            tieOdds=3.5,
            externalId="ext1",
            sport="soccer",
            league="Premier League",
            createDate="2023-01-01T00:00:00Z",
            lastUpdateDate="2023-01-01T00:00:00Z",
            eventStartDate="2023-01-01T00:00:00Z",
            active=True,
            outcome="unfinished",
            canTie=True,
        ),
        "game2": TeamGame(
            id="game2",
            teamA="Arsenal",
            teamB="Chelsea",
            teamAodds=2.5,
            teamBodds=2.8,
            tieOdds=3.2,
            externalId="ext2",
            sport="soccer",
            league="Premier League",
            createDate="2023-01-01T00:00:00Z",
            lastUpdateDate="2023-01-01T00:00:00Z",
            eventStartDate="2023-01-01T00:00:00Z",
            active=True,
            outcome="unfinished",
            canTie=True,
        ),
        "game3": TeamGame(
            id="game3",
            teamA="Real Madrid",
            teamB="Barcelona",
            teamAodds=1.8,
            teamBodds=4.0,
            tieOdds=3.0,
            externalId="ext3",
            sport="soccer",
            league="La Liga",
            createDate="2023-01-01T00:00:00Z",
            lastUpdateDate="2023-01-01T00:00:00Z",
            eventStartDate="2023-01-01T00:00:00Z",
            active=True,
            outcome="unfinished",
            canTie=True,
        ),
    }
    return games


def test_process_model_predictions_with_mock_data(mock_predictions_handler):
    sample_games = fetch_mock_games()

    predictions = mock_predictions_handler.process_model_predictions(
        sample_games, "soccer"
    )

    print("Model Predictions:")
    for game_id, prediction in predictions.items():
        print(f"Game: {game_id}")
        print(f"Predicted Outcome: {prediction.predictedOutcome}")
        print(f"Wager: {prediction.wager}")
        print(
            f"Odds: {prediction.teamAodds}, {prediction.teamBodds}, {prediction.tieOdds}"
        )
        print("---")

    print("Predictions after add_prediction:")
    for game_id, prediction in predictions.items():
        mock_predictions_handler.add_prediction(prediction.__dict__)
        print(f"Game: {game_id}")
        print(f"Prediction added: {prediction}")


if __name__ == "__main__":
    pytest.main()
