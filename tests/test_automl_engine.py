"""
AutoML Motoru, Tahmin ve Model Değerlendirme Testleri
"""

import numpy as np
import pandas as pd
import pytest
from modules.automl_engine import AutoMLEngine


def test_automl_classification(clean_sales_df):
    df = clean_sales_df.dropna().copy()
    features = ["Bölge", "Kategori", "Satış Adedi", "Birim Fiyat (TL)", "İndirim Oranı"]
    target = "Sadık Müşteri"

    result = AutoMLEngine.train_models(
        df=df,
        target_col=target,
        task_type="Sınıflandırma (Classification)",
        selected_features=features,
        test_size=0.2
    )

    assert "leaderboard_df" in result
    assert "best_model_name" in result
    assert "best_pipe" in result
    assert len(result["leaderboard_df"]) > 0

    best_m = result["best_model_name"]
    assert best_m is not None

    # Tekil tahmin testi
    sample_input = {
        "Bölge": "Marmara",
        "Kategori": "Elektronik",
        "Satış Adedi": 10,
        "Birim Fiyat (TL)": 500.0,
        "İndirim Oranı": 0.1
    }
    pred_val, prob_df, err = AutoMLEngine.predict_single(
        best_pipe=result["best_pipe"],
        raw_inputs=sample_input,
        engineered_features=result["engineered_features"],
        is_classification=True
    )

    assert err is None
    assert pred_val in ["Evet", "Hayır"]
    if prob_df is not None:
        assert len(prob_df) == 2


def test_automl_regression(clean_sales_df):
    df = clean_sales_df.dropna().copy()
    features = ["Bölge", "Kategori", "Satış Adedi", "Birim Fiyat (TL)", "İndirim Oranı"]
    target = "Toplam Gelir (TL)"

    result = AutoMLEngine.train_models(
        df=df,
        target_col=target,
        task_type="Regresyon (Regression)",
        selected_features=features,
        test_size=0.2
    )

    assert "leaderboard_df" in result
    lead_df = result["leaderboard_df"]
    assert "R² Skoru" in lead_df.columns
    assert lead_df.iloc[0]["R² Skoru"] > 0.5  # Gelir doğrusal ilişkili olduğu için yüksek R2 çıkmalı

    # Toplu tahmin testi
    test_batch_df = df[features].head(10).copy()
    batch_out, batch_err = AutoMLEngine.predict_batch(
        best_pipe=result["best_pipe"],
        target_batch_df=test_batch_df,
        selected_features=features,
        engineered_features=result["engineered_features"]
    )

    assert batch_err is None
    assert "Tahmin_Sonucu" in batch_out.columns
    assert len(batch_out) == 10


def test_feature_transformation_dates_and_ids():
    df = pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(50)],
        "Tarih": pd.date_range("2024-01-01", periods=50, freq="D"),
        "Deger": np.random.randn(50)
    })

    X_trans, dropped = AutoMLEngine.transform_features_for_ml(df)
    assert "ID" in dropped
    assert "Tarih_Ay" in X_trans.columns
    assert "Tarih_Gun" in X_trans.columns
    assert "Tarih_HaftaninGunu" in X_trans.columns
