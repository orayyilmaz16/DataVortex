"""
Şema Güvenliği, PyArrow Serileştirme & Dayanıklılık Testleri
"""

import numpy as np
import pandas as pd
import pytest
from modules.validator import SchemaValidator
from modules.data_processor import DataProcessor


def test_pyarrow_compatibility_and_mixed_types():
    # PyArrow hatası veren karma tip sütun oluştur
    df = pd.DataFrame({
        "Mixed_Col": [10, "yirmi", 30.5, b"kirk", None],
        "Bytes_Col": [b"elma", b"armut", b"muz", b"cilek", None]
    })

    sanitized = DataProcessor.sanitize_dataframe(df)
    # Serileştirme testi: Arrow uyumlu tabloya çevrilebilmeli
    try:
        import pyarrow as pa
        table = pa.Table.from_pandas(sanitized)
        assert table is not None
        assert table.num_rows == 5
    except ImportError:
        pass


def test_special_char_and_sql_injection_headers():
    df = pd.DataFrame({
        "Normal_Col": [1, 2],
        "Col with spaces": [3, 4],
        "Col; DROP TABLE users;--": [5, 6],
        "<img src=x onerror=alert(1)>": [7, 8]
    })

    clean_cols, issues = SchemaValidator.sanitize_column_names(df.columns.tolist())
    for col in clean_cols:
        assert ";" not in col or "DROP TABLE" not in col
        assert "<img" not in col


def test_empty_dataframe_resilience():
    empty_df = pd.DataFrame()
    res = DataProcessor.sanitize_dataframe(empty_df)
    assert res.empty
    report = SchemaValidator.validate_dataframe(empty_df)
    assert not report.is_valid
    assert report.score == 0.0
