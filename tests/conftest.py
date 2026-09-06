from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine

from forecasting_service.db import sqlalchemy_url
from forecasting_service.db_schema import metadata

ADMIN_DATABASE_URL = "postgresql://localhost:5432/symmetrical-octo-waddle"


@pytest.fixture
def database_url() -> Iterator[str]:
    schema = f"test_{uuid4().hex}"
    with psycopg.connect(ADMIN_DATABASE_URL, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    url = f"{ADMIN_DATABASE_URL}?options=-csearch_path%3D{schema}"
    engine = create_engine(sqlalchemy_url(url))
    metadata.create_all(engine)
    engine.dispose()
    try:
        yield url
    finally:
        with psycopg.connect(ADMIN_DATABASE_URL, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
