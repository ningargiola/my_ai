import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import Base, engine, get_db
from sqlalchemy.orm import Session

@pytest.fixture(scope="session")
def test_db():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up after tests
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(test_db):
    with TestClient(app) as test_client:
        yield test_client 