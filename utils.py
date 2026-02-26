# utils.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame, returning an empty frame on failure."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {len(df):,} rows from '{path}'")
        return df
    except FileNotFoundError:
        print(f"⚠️  File not found: '{path}' — returning empty DataFrame")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️  Error loading '{path}': {e} — returning empty DataFrame")
        return pd.DataFrame()