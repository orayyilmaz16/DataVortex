"""
DataVortex — Streamlit Caching Katmanı (Cached Adapters)
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st


def get_cached_sample_data(loader_func: Any) -> pd.DataFrame:
    """Örnek veri setini hafızada önbelleğe alır ve döner."""
    @st.cache_data(show_spinner=False)
    def _cached_sample() -> pd.DataFrame:
        return loader_func()
    return _cached_sample()


@st.cache_data(show_spinner=False)
def load_excel_cached(file_bytes: bytes, sheet_name: Optional[str], header_row: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Excel tablosunu hafızadaki bayt akışından hızlı ve hatasız yükler."""
    from datavortex.data.io import load_excel
    return load_excel(file_bytes, sheet_name=sheet_name, header=header_row)


@st.cache_data(show_spinner=False)
def load_csv_cached(
    file_bytes: bytes,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    decimal: Optional[str] = None,
    header_row: int = 0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """CSV tablosunu akıllı kodlama, ayraç ve ondalık tespitiyle hatasız yükler."""
    from datavortex.data.io import load_csv
    return load_csv(
        file_bytes=file_bytes,
        sep=sep,
        encoding=encoding,
        decimal=decimal,
        header=header_row
    )
