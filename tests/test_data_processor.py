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


def test_detect_encoding_and_delimiter():
    csv_str = "Ürün Adı;Kategori;Fiyat;Satış Adedi\nBilgisayar;Elektronik;25000,50;10\nŞapka;Giyim;150,00;45\nÇanta;Aksesuar;450,75;20"
    b_utf8 = csv_str.encode("utf-8")
    enc = DataProcessor.detect_encoding(b_utf8)
    assert enc in ["utf-8", "utf-8-sig"]
    delim = DataProcessor.detect_delimiter(csv_str)
    assert delim == ";"

    b_cp1254 = csv_str.encode("cp1254")
    enc_cp = DataProcessor.detect_encoding(b_cp1254)
    assert enc_cp in ["cp1254", "iso-8859-9"]

    b_bom = b"\xef\xbb\xbf" + csv_str.encode("utf-8")
    assert DataProcessor.detect_encoding(b_bom) == "utf-8-sig"


def test_load_csv_automatic():
    csv_text = "Müşteri Adı;Şehir;Harcama Tutarı;Sadık Mı\nAhmet Çelik;İstanbul;1.250,50;Evet\nAyşe Güner;İzmir;850,25;Hayır\nÖmer Şen;Ankara;3.400,00;Evet\n"
    file_bytes = csv_text.encode("utf-8-sig")

    df, meta = DataProcessor.load_csv(file_bytes)
    assert len(df) == 3
    assert len(df.columns) == 4
    assert meta["delimiter"] == ";"
    assert "Müşteri Adı" in df.columns
    assert "Sadık Mı" in df.columns
    assert pd.api.types.is_numeric_dtype(df["Harcama Tutarı"])
    assert df["Harcama Tutarı"].iloc[0] == pytest.approx(1250.50)


def test_load_csv_cp1254():
    csv_text = "Bölge,Ürün,Gelir\nİç Anadolu,Güneş Gözlüğü,450.50\nEge,Şort,120.00\n"
    file_bytes = csv_text.encode("cp1254")

    df, meta = DataProcessor.load_csv(file_bytes)
    assert len(df) == 2
    assert "Bölge" in df.columns
    assert df["Bölge"].iloc[0] == "İç Anadolu"
    assert df["Ürün"].iloc[0] == "Güneş Gözlüğü"


def test_load_file_unified():
    tsv_text = "ID\tDeger\tSkor\n1\tA\t10.5\n2\tB\t20.2\n"
    df, meta = DataProcessor.load_file(tsv_text.encode("utf-8"), "test.tsv")
    assert len(df) == 2
    assert meta["delimiter"] == "\t"

    empty_df, meta_empty = DataProcessor.load_file(b"", "empty.csv")
    assert empty_df.empty
    assert meta_empty["rows"] == 0
