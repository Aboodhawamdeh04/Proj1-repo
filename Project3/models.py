from sqlalchemy import MetaData, Table, Column, Integer, String, LargeBinary

metadata = MetaData()

destinations_table = Table(
    "destinations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("country", String, nullable=False),
    Column("description", String, nullable=False),
    Column("category", String, nullable=False),
    Column("embedding", LargeBinary, nullable=False)
)