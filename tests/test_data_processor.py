"""
Veri Manipülasyonu, Temizleme & İşlem Geçmişi (Undo/Redo) Testleri
"""

import numpy as np
import pandas as pd
import pytest
from modules.data_processor import DataHistoryManager, DataProcessor


def test_sanitize_dataframe(edge_case_df):
    clean_df = DataProcessor.sanitize_dataframe(edge_case_df)
    assert not clean_df.empty
    # Sütun isimleri temizlenmiş olmalı
    assert not any("<script>" in str(c) for c in clean_df.columns)
    # PyArrow nesne tipleri homojen olmalı
    for c in clean_df.columns:
        assert clean_df[c].dtype != "object" or all(isinstance(x, str) for x in clean_df[c].dropna())


def test_undo_redo_manager(clean_sales_df):
    history = DataHistoryManager(max_history=5)
    df_init = clean_sales_df.copy()

    assert not history.can_undo()
    assert not history.can_redo()

    # 1. İşlem: Bir sütun sil
    df_step1 = df_init.drop(columns=["Ödeme Yöntemi"])
    history.record_change(df_init, df_step1, "Sütun Sil", "Ödeme Yöntemi silindi")

    assert history.can_undo()
    assert not history.can_redo()
    assert len(history.undo_stack) == 1

    # 2. İşlem: Satır filtrele
    df_step2 = df_step1.head(200)
    history.record_change(df_step1, df_step2, "Satır Filtrele", "İlk 200 satır alındı")

    assert len(history.undo_stack) == 2

    # Geri Al (Undo step 2 -> step 1)
    df_undone1, rec1 = history.undo(df_step2)
    assert len(df_undone1) == len(df_step1)
    assert history.can_redo()
    assert "Satır Filtrele" in rec1.action_name

    # Geri Al (Undo step 1 -> init)
    df_undone2, rec2 = history.undo(df_undone1)
    assert "Ödeme Yöntemi" in df_undone2.columns
    assert not history.can_undo()

    # İleri Al (Redo init -> step 1)
    df_redone1, rec_r1 = history.redo(df_undone2)
    assert "Ödeme Yöntemi" not in df_redone1.columns
    assert history.can_undo()


def test_imputation_methods(messy_df):
    # Fiyat sütununda eksikler var
    assert messy_df["Fiyat"].isna().sum() > 0

    # Ortalama ile doldur
    imputed_df, msg = DataProcessor.apply_imputation(messy_df, "Fiyat", "Ortalama ile Doldur (Mean)")
    assert imputed_df["Fiyat"].isna().sum() == 0
    assert "başarıyla dolduruldu" in msg

    # Eksik içeren satırları sil
    dropped_df, drop_msg = DataProcessor.apply_imputation(messy_df, "Kategori", "Eksik İçeren Satırları Sil (Drop Rows)")
    assert dropped_df["Kategori"].isna().sum() == 0
    assert len(dropped_df) < len(messy_df)


def test_outlier_clipping(clean_sales_df):
    df = clean_sales_df.copy()
    col = "Toplam Gelir (TL)"

    # Manuel aşırı aykırı ekle
    df.loc[0, col] = 999999.0

    clipped_df, count, msg = DataProcessor.apply_outlier_action(
        df, col, method="IQR (Çeyrekler Açıklığı)", action="Sınır Değerlere Kırp (Clip)"
    )

    assert count >= 1
    assert clipped_df[col].max() < 999999.0
    assert "kırpıldı" in msg


def test_calculated_column(clean_sales_df):
    df = clean_sales_df.copy()
    new_df, err = DataProcessor.create_calculated_column(
        df,
        new_col_name="Birim_Kar",
        formula_str="[Net Kâr (TL)] / [Satış Adedi]"
    )
    assert err is None
    assert "Birim_Kar" in new_df.columns
    assert not new_df["Birim_Kar"].isna().all()


def test_filter_rule(clean_sales_df):
    df = clean_sales_df.copy()
    filtered_df, dropped_cnt, err = DataProcessor.apply_filter_rule(
        df, col="Satış Adedi", operator="Büyüktür (>)", val1=20
    )
    assert err is None
    assert (filtered_df["Satış Adedi"] > 20).all()
    assert dropped_cnt == len(df) - len(filtered_df)
