"""
Validasyon ve Güvenlik Motoru Birim Testleri
"""

import numpy as np
import pandas as pd
import pytest
from modules.validator import (
    DataStressTester,
    FormulaValidator,
    InputValidator,
    MLGuard,
    SchemaValidator,
    ValidationSeverity,
)


def test_schema_validator_empty_and_none():
    rep_none = SchemaValidator.validate_dataframe(None)
    assert not rep_none.is_valid
    assert any(i.code == "DATAFRAME_NONE" for i in rep_none.issues)

    rep_empty = SchemaValidator.validate_dataframe(pd.DataFrame())
    assert not rep_empty.is_valid
    assert any(i.code == "DATAFRAME_EMPTY" for i in rep_empty.issues)


def test_sanitize_column_names():
    raw_cols = ["  Ad Soyad  ", "Unnamed: 0", "Unnamed: 1", "<script>bad</script>", "Gelir", "Gelir"]
    clean_cols, issues = SchemaValidator.sanitize_column_names(raw_cols)

    assert clean_cols[0] == "Ad Soyad"
    assert clean_cols[1] == "Sütun_2"
    assert clean_cols[2] == "Sütun_3"
    assert "<script>" not in clean_cols[3]
    assert clean_cols[4] == "Gelir"
    assert clean_cols[5] == "Gelir_1"
    assert len(issues) >= 3


def test_schema_validator_infinite_and_zero_variance(edge_case_df):
    report = SchemaValidator.validate_dataframe(edge_case_df)
    assert any(i.code == "INFINITE_VALUES_DETECTED" for i in report.issues)
    assert any(i.code == "ZERO_VARIANCE_COLUMN" for i in report.issues)


def test_input_validator_imputation():
    s_num = pd.Series([10.0, 20.0, np.nan, 40.0])
    s_cat = pd.Series(["A", "B", np.nan, "A"])

    # Sayısal sütunda ortalama doldurma geçerli olmalı
    rep1 = InputValidator.validate_imputation(s_num, "Ortalama ile Doldur (Mean)")
    assert rep1.is_valid

    # Metin sütununda ortalama doldurma hata vermeli
    rep2 = InputValidator.validate_imputation(s_cat, "Ortalama ile Doldur (Mean)")
    assert not rep2.is_valid
    assert any(i.code == "INVALID_NUMERIC_METHOD_ON_TEXT" for i in rep2.issues)

    # Sayısal sütuna geçersiz metin sabit değer girilirse hata vermeli
    rep3 = InputValidator.validate_imputation(s_num, "Sabit Değer ile Doldur", custom_val="gecersiz_sayi")
    assert not rep3.is_valid
    assert any(i.code == "TYPE_MISMATCH_NUMERIC_CONSTANT" for i in rep3.issues)


def test_input_validator_outlier_bounds():
    s = pd.Series([10, 20, 30, 40, 50])
    rep_bad = InputValidator.validate_outlier_bounds(50.0, 20.0, s)
    assert not rep_bad.is_valid
    assert any(i.code == "INVALID_BOUND_RANGE" for i in rep_bad.issues)

    rep_good = InputValidator.validate_outlier_bounds(10.0, 50.0, s)
    assert rep_good.is_valid


def test_formula_validator_safe_evaluation():
    df = pd.DataFrame({
        "Satis": [100, 200, 300],
        "Fiyat": [10.0, 20.0, 30.0],
        "Oran": [0.1, 0.2, 0.3]
    })

    # Geçerli aritmetik formül
    res, err = FormulaValidator.evaluate_safe_formula(df, "[Satis] * [Fiyat] * (1 - [Oran])")
    assert err is None
    assert res is not None
    assert len(res) == 3
    assert np.isclose(res.iloc[0], 100 * 10.0 * 0.9)

    # Matematik fonksiyonu kullanımı
    res_log, err_log = FormulaValidator.evaluate_safe_formula(df, "log([Fiyat]) + 5")
    assert err_log is None
    assert res_log is not None

    # Güvensiz / Tehlikeli kod engellenmeli
    _, err_hack = FormulaValidator.evaluate_safe_formula(df, "__import__('os').system('dir')")
    assert err_hack is not None

    # Olmayan sütun başvurusu engellenmeli
    _, err_col = FormulaValidator.evaluate_safe_formula(df, "[Olmayan_Sutun] + 10")
    assert err_col is not None


def test_ml_guard_validation(clean_sales_df):
    df = clean_sales_df

    # Geçerli Sınıflandırma kurulumu
    rep1 = MLGuard.validate_ml_setup(
        df=df,
        target_col="Sadık Müşteri",
        task_type="Sınıflandırma (Classification)",
        selected_features=["Bölge", "Kategori", "Satış Adedi", "Toplam Gelir (TL)"]
    )
    assert rep1.is_valid

    # Tarih sütunu hedef olamaz
    rep_date = MLGuard.validate_ml_setup(
        df=df,
        target_col="Tarih",
        task_type="Sınıflandırma (Classification)",
        selected_features=["Bölge", "Kategori"]
    )
    assert not rep_date.is_valid
    assert any(i.code == "DATE_TARGET_NOT_ALLOWED" for i in rep_date.issues)

    # Metin sütunu regresyon hedefi olamaz
    rep_reg_text = MLGuard.validate_ml_setup(
        df=df,
        target_col="Bölge",
        task_type="Regresyon (Regression)",
        selected_features=["Satış Adedi", "Toplam Gelir (TL)"]
    )
    assert not rep_reg_text.is_valid
    assert any(i.code == "NON_NUMERIC_REGRESSION_TARGET" for i in rep_reg_text.issues)


def test_data_stress_tester(clean_sales_df):
    diag = DataStressTester.run_full_diagnostics(clean_sales_df)
    assert diag["overall_score"] >= 80.0
    assert diag["tests_passed"] >= 8
    assert len(diag["details"]) == diag["total_tests"]
