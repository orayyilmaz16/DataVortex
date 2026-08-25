"""
Ortak Test Fikstürleri ve Sentetik Veri Üreteçleri
"""

import pytest
import numpy as np
import pandas as pd
from modules.data_processor import DataProcessor


@pytest.fixture
def clean_sales_df() -> pd.DataFrame:
    """Temiz ve eksiksiz örnek satış veri seti."""
    return DataProcessor.generate_sample_sales_data()


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Eksik değerler, aykırılıklar ve mükerrer satırlar içeren kirli veri seti."""
    np.random.seed(123)
    n = 100
    df = pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(n)],
        "Tarih": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Kategori": np.random.choice(["A", "B", "C"], n),
        "Fiyat": np.random.uniform(50.0, 500.0, n),
        "Adet": np.random.randint(1, 20, n),
        "Durum": np.random.choice(["Aktif", "Pasif"], n)
    })
    # Eksik değerler ekle
    df.loc[0:5, "Fiyat"] = np.nan
    df.loc[10:12, "Kategori"] = np.nan
    # Aykırı değer ekle
    df.loc[20, "Fiyat"] = 99999.0
    # Duplicate satır ekle
    df = pd.concat([df, df.iloc[0:3]], ignore_index=True)
    return df


@pytest.fixture
def edge_case_df() -> pd.DataFrame:
    """Sıfır varyans, sonsuz değerler ve karma tipler içeren sınır durum veri seti."""
    return pd.DataFrame({
        "Sabit_Sutun": [42] * 20,
        "Sonsuz_Sutun": [1.0, 2.0, np.inf, 4.0, -np.inf] + [3.0] * 15,
        "Karma_Sutun": [1, "iki", 3.0, b"dort", 5] + ["altı"] * 15,
        "<script>alert(1)</script>": range(20)
    })
