import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session
from bettensor.miner.database.database_manager import DatabaseManager


@pytest.fixture
def mock_db_session():
    return MagicMock(spec=Session)


@pytest.fixture
def mock_db_manager(mock_db_session):
    with patch("bettensor.miner.database.database_manager.create_engine"), patch(
        "bettensor.miner.database.database_manager.sessionmaker"
    ) as mock_sessionmaker:
        mock_sessionmaker.return_value = lambda: mock_db_session
        yield DatabaseManager()
