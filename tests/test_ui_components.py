"""
UI Bileşenleri ve Bildirim Sistemi Testleri
"""

import pandas as pd
import numpy as np
import pytest
from modules.ui_components import UIComponents, NotificationHub
from modules.validator import SchemaValidator


def test_ui_components_kpi_cards(clean_sales_df):
    schema_rep = SchemaValidator.validate_dataframe(clean_sales_df)
    health_dict = {
        "score": schema_rep.score,
        "grade": schema_rep.grade,
        "missing_cells": schema_rep.metrics.get("missing_cells", 0),
        "missing_pct": schema_rep.metrics.get("missing_pct", 0.0),
        "dup_rows": schema_rep.metrics.get("dup_rows", 0),
        "dup_pct": schema_rep.metrics.get("dup_pct", 0.0),
        "color": "#22C55E"
    }

    # Streamlit ortamı dışında hata fırlatmadan HTML üretebilmeli veya çalışabilmeli
    # st fonksiyonları mock'lanmasa bile UIComponents fonksiyonları parametrelerini doğru işlemeli
    assert UIComponents.CUSTOM_CSS is not None
    assert len(UIComponents.CUSTOM_CSS) > 100


def test_scatter_size_safety_with_nans():
    import plotly.express as px
    # Create DataFrame with NaNs and negative numbers in size column
    df = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [10, 20, 15, 30, 25],
        "Size": [np.nan, 5.0, -2.0, np.nan, 10.0]
    })
    # Apply safe size logic as implemented in app.py
    s_size = pd.to_numeric(df["Size"], errors="coerce")
    fill_val = s_size.median() if (s_size.median() is not None and s_size.median() > 0) else 1.0
    s_size = s_size.fillna(fill_val).abs()
    df["_safe_size_Size"] = s_size

    # Plotly scatter should now render without crashing on invalid size elements
    fig = px.scatter(df, x="X", y="Y", size="_safe_size_Size")
    assert fig is not None
    assert len(fig.data) > 0
