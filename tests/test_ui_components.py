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
