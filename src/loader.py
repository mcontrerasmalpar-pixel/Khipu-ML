"""
loader.py — SQLite connection and data loading for the OKR database.
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "khipu.db"


def get_engine(db_path: Path = DEFAULT_DB_PATH):
    """Return a SQLAlchemy engine connected to the OKR SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Download khipu.db from https://doi.org/10.5281/zenodo.18025748 "
            "and place it in the data/ directory."
        )
    return create_engine(f"sqlite:///{db_path}")


def list_tables(engine=None) -> list[str]:
    """Return a list of all table names in the database."""
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        return [row[0] for row in result]


def load_table(table: str, engine=None) -> pd.DataFrame:
    """Load an entire table into a DataFrame."""
    if engine is None:
        engine = get_engine()
    return pd.read_sql_table(table, engine)


def load_query(sql: str, engine=None) -> pd.DataFrame:
    """Execute a raw SQL query and return the result as a DataFrame."""
    if engine is None:
        engine = get_engine()
    return pd.read_sql_query(sql, engine)
