import argparse
from typing import Any

from sqlalchemy import create_engine, inspect

from forecasting_service.config import Settings
from forecasting_service.db_schema import COLLECTIONS, metadata


def sqlalchemy_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def schema_status(engine: Any) -> tuple[list[str], list[str]]:
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    expected = set(COLLECTIONS)
    missing = sorted(expected - existing)
    incompatible = []
    for table in sorted(expected & existing):
        columns = {column["name"] for column in inspector.get_columns(table)}
        expected_columns = {column.name for column in metadata.tables[table].columns}
        if columns != expected_columns:
            incompatible.append(table)
    return missing, incompatible


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the forecasting metadata database")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("push", help="create missing schema objects without deleting data")
    subcommands.add_parser("sync", help="push schema and verify table compatibility")
    subcommands.add_parser("status", help="show schema compatibility")
    args = parser.parse_args()
    database_url = Settings().database_url
    if not database_url:
        parser.error("DATABASE_URL is required for database commands")
    engine = create_engine(sqlalchemy_url(database_url))

    if args.command in {"push", "sync"}:
        metadata.create_all(engine, checkfirst=True)
    missing, incompatible = schema_status(engine)
    engine.dispose()
    if missing or incompatible:
        details = [f"missing={missing}", f"incompatible={incompatible}"]
        raise SystemExit("schema mismatch: " + ", ".join(details))
    print(f"schema ready: {len(COLLECTIONS)} resource tables")
