from sqlalchemy import Column, DateTime, Index, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB

INDEXED_FIELDS: dict[str, tuple[str, ...]] = {
    "datasets": ("state",),
    "experiments": ("dataset_id", "state"),
    "jobs": ("resource_id", "state"),
    "models": ("dataset_id", "state", "stage"),
    "forecasts": ("model_id", "state"),
    "actuals": ("model_id", "state"),
    "idempotency": ("scope", "resource_id", "fingerprint"),
}

COLLECTIONS = tuple(INDEXED_FIELDS)

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "pk": "pk_%(table_name)s",
    }
)


def resource_table(name: str, fields: tuple[str, ...]) -> Table:
    table = Table(
        name,
        metadata,
        Column("id", String(80), primary_key=True),
        Column("tenant_id", String(100), nullable=True),
        *(Column(field, String(100), nullable=True) for field in fields),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("payload", JSONB, nullable=False),
    )
    Index(f"ix_{name}_tenant_created", table.c.tenant_id, table.c.created_at.desc())
    for field in fields:
        Index(f"ix_{name}_{field}", table.c[field])
    return table


tables = {name: resource_table(name, fields) for name, fields in INDEXED_FIELDS.items()}
