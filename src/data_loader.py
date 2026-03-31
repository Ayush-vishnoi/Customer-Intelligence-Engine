"""
data_loader.py
--------------
Loads the raw Online Retail Excel file, applies all cleaning steps,
and returns a tidy DataFrame ready for feature engineering.

Cleaning steps
--------------
1. Drop rows with missing CustomerID
2. Remove cancelled invoices  (InvoiceNo starts with 'C')
3. Remove negative / zero Quantity and UnitPrice
4. Drop exact duplicate rows
5. Parse InvoiceDate to datetime
6. Derive TotalPrice = Quantity × UnitPrice
"""

import pandas as pd


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def load_and_clean(path: str = "data/Online_Retail.xlsx") -> pd.DataFrame:
    """
    Load and clean the Online Retail dataset.

    Parameters
    ----------
    path : str
        Path to the raw .xlsx file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an added TotalPrice column.
    """
    print(f"[data_loader] Loading: {path}")
    df = pd.read_excel(path)
    print(f"[data_loader] Raw shape: {df.shape}")

    df = _drop_missing_customers(df)
    df = _remove_cancellations(df)
    df = _remove_invalid_quantities(df)
    df = _drop_duplicates(df)
    df = _parse_dates(df)
    df = _add_total_price(df)

    print(f"[data_loader] Clean shape: {df.shape}")
    print(f"[data_loader] Customers: {df['CustomerID'].nunique():,}")
    print(f"[data_loader] Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    return df


def get_summary(df: pd.DataFrame) -> dict:
    """Return a quick stats dictionary for reporting."""
    return {
        "n_rows":      len(df),
        "n_customers": df["CustomerID"].nunique(),
        "n_products":  df["StockCode"].nunique(),
        "n_countries": df["Country"].nunique(),
        "date_min":    df["InvoiceDate"].min().date(),
        "date_max":    df["InvoiceDate"].max().date(),
        "total_revenue": df["TotalPrice"].sum(),
    }


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _drop_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    print(f"[data_loader] Dropped {before - len(df):,} rows with missing CustomerID")
    return df


def _remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~mask].copy()
    print(f"[data_loader] Removed {before - len(df):,} cancelled invoices")
    return df


def _remove_invalid_quantities(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    print(f"[data_loader] Removed {before - len(df):,} rows with invalid Quantity/UnitPrice")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"[data_loader] Dropped {before - len(df):,} duplicate rows")
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def _add_total_price(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df
