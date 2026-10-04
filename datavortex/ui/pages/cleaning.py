"""
DataVortex — Sayfa 2: Veri Temizleme, Manipülasyon ve Geri Alma (Undo/Redo)
"""

import time
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st

from datavortex.core.state import StateKeys, clear_automl_result
from datavortex.data.cleaning import apply_imputation, apply_outlier_action
from datavortex.data.history import DataHistoryManager
from datavortex.data.io import sanitize_dataframe
from datavortex.data.transforms import (
    apply_filter_rule,
    apply_type_cast,
    create_calculated_column,
)
from datavortex.ui.components import NotificationHub
from datavortex.validation.input import InputValidator


def render_cleaning_page(df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """Veri temizleme, dönüştürme ve geçmiş yönetim arayüzünü render eder."""
    st.markdown("### 🧹 Akıllı Veri Temizleme & Manipülasyon Stüdyosu")
    st.info("💡 Burada yaptığınız tüm işlemler anlık olarak aktif veri setine yansır ve işlem geçmişinde saklanır. İstediğiniz zaman geri alabilirsiniz.")

    history_mgr = st.session_state.get(StateKeys.HISTORY_MGR)
    if history_mgr is None:
        history_mgr = DataHistoryManager(max_history=15)
        st.session_state[StateKeys.HISTORY_MGR] = history_mgr

    col_undo1, col_undo2, col_undo3 = st.columns([1, 1, 3])

    with col_undo1:
        if st.button("↩️ Son İşlemi Geri Al (Undo)", disabled=not history_mgr.can_undo(), width="stretch"):
            with st.spinner("Geri alınıyor..."):
                start_t = time.time()
                prev_df, rec = history_mgr.undo(df)
                st.session_state[StateKeys.ACTIVE_DF] = prev_df
                duration = (time.time() - start_t) * 1000
                st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                    "action_name": "Geri Alma (Undo)",
                    "description": f"'{rec.action_name}' işlemi geri alındı.",
                    "duration_ms": duration
                }
                NotificationHub.notify_info(f"İşlem geri alındı: {rec.action_name}")
                st.rerun()

    with col_undo2:
        if st.button("↪️ İleri Al (Redo)", disabled=not history_mgr.can_redo(), width="stretch"):
            with st.spinner("İleri alınıyor..."):
                start_t = time.time()
                next_df, rec = history_mgr.redo(df)
                st.session_state[StateKeys.ACTIVE_DF] = next_df
                duration = (time.time() - start_t) * 1000
                st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                    "action_name": "İleri Alma (Redo)",
                    "description": f"'{rec.action_name}' tekrar uygulandı.",
                    "duration_ms": duration
                }
                NotificationHub.notify_info(f"İşlem ileri alındı: {rec.action_name}")
                st.rerun()

    with col_undo3:
        with st.expander("📜 İşlem Geçmişi (History Timeline)", expanded=False):
            hist_list = history_mgr.get_history_list()
            if hist_list:
                st.dataframe(pd.DataFrame(hist_list), width="stretch")
            else:
                st.caption("Henüz bir manipülasyon işlemi yapılmadı.")

    st.markdown("---")
    col_clean_a, col_clean_b = st.columns(2)

    with col_clean_a:
        # 1. Eksik Değer Yönetimi
        with st.expander("🚨 1. Eksik Değer (Missing Value) Yönetimi", expanded=True):
            missing_cols = df.columns[df.isna().any()].tolist()
            if missing_cols:
                st.write(f"Eksik değer içeren sütunlar: `{'`, `'.join(missing_cols)}`")
                clean_target_col = st.selectbox("İşlem Yapılacak Sütun", ["Tüm Sayısal Sütunlar", "Tüm Kategorik Sütunlar"] + missing_cols)
                impute_method = st.selectbox(
                    "Doldurma / Temizleme Yöntemi",
                    [
                        "Ortalama ile Doldur (Mean)",
                        "Medyan ile Doldur (Median)",
                        "En Çok Tekrar Eden ile Doldur (Mode)",
                        "Sabit Değer ile Doldur",
                        "Önceki Değerle Doldur (Forward Fill)",
                        "Sonraki Değerle Doldur (Backward Fill)",
                        "Eksik İçeren Satırları Sil (Drop Rows)",
                        "Eksik İçeren Sütunu Tamamen Sil (Drop Column)"
                    ]
                )
                custom_fill_val = None
                if impute_method == "Sabit Değer ile Doldur":
                    custom_fill_val = st.text_input("Doldurulacak Sabit Değer:", value="0")

                if st.button("✨ Eksik Veri İşlemini Uygula", width="stretch"):
                    if clean_target_col in df.columns:
                        val_rep = InputValidator.validate_imputation(df[clean_target_col], impute_method, custom_fill_val)
                        if not val_rep.is_valid:
                            for issue in val_rep.get_errors():
                                st.error(f"❌ {issue.message}")
                            st.stop()

                    start_t = time.time()
                    with st.spinner("Eksik değerler dolduruluyor..."):
                        old_state = df.copy()
                        new_df, desc_msg = apply_imputation(df, clean_target_col, impute_method, custom_fill_val)
                        history_mgr.record_change(old_state, new_df, "Eksik Değer Doldurma", desc_msg)
                        st.session_state[StateKeys.ACTIVE_DF] = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                            "action_name": "Eksik Değer Yönetimi",
                            "description": desc_msg,
                            "affected_rows": abs(len(old_state) - len(new_df)),
                            "duration_ms": duration
                        }
                        NotificationHub.notify_success(desc_msg)
                        st.rerun()
            else:
                st.success("🎉 Tebrikler! Veri setinizde hiçbir eksik değer bulunmuyor.")

        # 2. Mükerrer Kayıt Temizliği
        with st.expander("🔁 2. Mükerrer (Duplicate) Kayıt Temizliği", expanded=False):
            dup_count = df.duplicated().sum()
            st.write(f"Tespit Edilen Duplicate Satır Sayısı: **{dup_count}**")
            dup_subset = st.multiselect("Kontrol Edilecek Özel Sütunlar (Boş bırakılırsa tüm sütunlar baz alınır):", df.columns)

            if st.button("🗑️ Duplicate Kayıtları Kaldır", width="stretch", disabled=(dup_count == 0)):
                start_t = time.time()
                with st.spinner("Mükerrer satırlar siliniyor..."):
                    old_state = df.copy()
                    subset = dup_subset if dup_subset else None
                    new_df = df.drop_duplicates(subset=subset)
                    dropped = len(df) - len(new_df)
                    new_df = sanitize_dataframe(new_df)

                    desc_msg = f"{dropped} adet mükerrer satır silindi."
                    history_mgr.record_change(old_state, new_df, "Duplicate Temizliği", desc_msg)
                    st.session_state[StateKeys.ACTIVE_DF] = new_df
                    duration = (time.time() - start_t) * 1000
                    st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                        "action_name": "Duplicate Temizliği",
                        "description": desc_msg,
                        "affected_rows": dropped,
                        "duration_ms": duration
                    }
                    NotificationHub.notify_success(desc_msg)
                    st.rerun()

        # 3. Gelişmiş Filtreleme Motoru
        with st.expander("🔍 3. Gelişmiş Kural Tabanlı Filtreleme", expanded=False):
            filter_col = st.selectbox("Filtrelenecek Sütun:", df.columns, key="filter_col_select")
            filter_op = st.selectbox(
                "Filtreleme Operatörü:",
                [
                    "Eşittir (==)", "Eşit Değildir (!=)", "Büyüktür (>)", "Küçüktür (<)",
                    "Büyük veya Eşittir (>=)", "Küçük veya Eşittir (<=)", "Aralığında (Between)",
                    "İçerir (Contains)", "İçermez (Not Contains)", "Boş Olanlar (Is Null)", "Dolu Olanlar (Is Not Null)"
                ]
            )
            f_val1 = None
            f_val2 = None
            if filter_op not in ["Boş Olanlar (Is Null)", "Dolu Olanlar (Is Not Null)"]:
                f_val1 = st.text_input("1. Değer:", value="", key="filter_v1")
                if filter_op == "Aralığında (Between)":
                    f_val2 = st.text_input("2. Değer (Üst Sınır):", value="", key="filter_v2")

            if st.button("⚡ Filtreyi Uygula ve Veriyi Güncelle", width="stretch"):
                start_t = time.time()
                with st.spinner("Filtre uygulanıyor..."):
                    old_state = df.copy()
                    new_df, dropped_rows, err = apply_filter_rule(df, filter_col, filter_op, f_val1, f_val2)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        desc_msg = f"'{filter_col}' {filter_op} filtresi uygulandı ({dropped_rows} satır elendi)."
                        history_mgr.record_change(old_state, new_df, "Kural Filtreleme", desc_msg)
                        st.session_state[StateKeys.ACTIVE_DF] = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                            "action_name": "Filtreleme",
                            "description": desc_msg,
                            "affected_rows": dropped_rows,
                            "duration_ms": duration
                        }
                        NotificationHub.notify_success(desc_msg)
                        st.rerun()

    with col_clean_b:
        # 4. Aykırı Değer Budama / Kırpma
        with st.expander("⚡ 4. Aykırı Değer (Outlier) Tespiti ve Kırpma", expanded=True):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                outlier_col = st.selectbox("Aykırı Değer Analizi Yapılacak Sütun:", numeric_cols)
                outlier_method = st.radio("Tespit Yöntemi", ["IQR (Çeyrekler Açıklığı - 1.5x IQR)", "Z-Score (> 3 Standart Sapma)"], horizontal=True)
                outlier_action = st.selectbox("Uygulanacak Eylem", ["Sınır Değerlere Kırp (Clip)", "Satırları Sil", "Aykırı Etiket Sütunu Ekle"])

                series = df[outlier_col].dropna()
                if not series.empty:
                    if st.button("✂️ Aykırı Değer İşlemini Uygula", width="stretch"):
                        start_t = time.time()
                        with st.spinner("Aykırı değerler işleniyor..."):
                            old_state = df.copy()
                            new_df, out_cnt, desc_msg = apply_outlier_action(
                                df, outlier_col, outlier_method, outlier_action
                            )
                            history_mgr.record_change(old_state, new_df, "Aykırı Değer Yönetimi", desc_msg)
                            st.session_state[StateKeys.ACTIVE_DF] = new_df
                            duration = (time.time() - start_t) * 1000
                            st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                                "action_name": "Aykırı Değer Yönetimi",
                                "description": desc_msg,
                                "affected_rows": out_cnt,
                                "duration_ms": duration
                            }
                            NotificationHub.notify_success(desc_msg)
                            st.rerun()
            else:
                st.warning("Aykırı değer analizi için sayısal sütun bulunamadı.")

        # 5. AST Tabanlı Güvenli Formül & Hesaplanan Sütun Türetme
        with st.expander("🧮 5. Formül ile Yeni Sütun Türet (Calculated Column)", expanded=False):
            st.caption("Güvenli matematiksel formüllerle yeni sütunlar oluşturun. Sütun isimlerini `[Sütun_Adı]` formatında yazın.")
            new_col_name = st.text_input("Yeni Sütun Adı:", value="Net_Birim_Fiyat")
            formula_input = st.text_input("Matematiksel Formül:", value="[Birim Fiyat (TL)] * (1 - [İndirim Oranı])")
            st.caption("Örnek fonksiyonlar: `abs()`, `log()`, `sqrt()`, `round()`, `exp()`, `sin()`, `cos()`")

            if st.button("🚀 Formülü Hesapla ve Yeni Sütun Ekle", width="stretch"):
                start_t = time.time()
                with st.spinner("Formül doğrulanıyor ve hesaplanıyor..."):
                    old_state = df.copy()
                    new_df, err = create_calculated_column(df, new_col_name, formula_input)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        desc_msg = f"'{new_col_name}' sütunu formülle başarıyla türetildi."
                        history_mgr.record_change(old_state, new_df, "Hesaplanan Sütun", desc_msg)
                        st.session_state[StateKeys.ACTIVE_DF] = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                            "action_name": "Hesaplanan Sütun",
                            "description": desc_msg,
                            "duration_ms": duration
                        }
                        NotificationHub.notify_success(desc_msg)
                        st.rerun()

        # 6. Sütun Yönetimi & Tip Dönüştürme
        with st.expander("🔧 6. Sütun Yönetimi & Tip Dönüştürme", expanded=False):
            col_manage_tab1, col_manage_tab2 = st.tabs(["Sütun Sil", "Tip Dönüştür"])
            with col_manage_tab1:
                cols_to_drop = st.multiselect("Silinecek Sütunları Seçin:", df.columns)
                if st.button("🗑️ Seçilen Sütunları Sil", width="stretch", disabled=not cols_to_drop):
                    start_t = time.time()
                    with st.spinner("Sütunlar siliniyor..."):
                        old_state = df.copy()
                        new_df = df.drop(columns=cols_to_drop)
                        new_df = sanitize_dataframe(new_df)
                        desc_msg = f"{len(cols_to_drop)} sütun silindi."
                        history_mgr.record_change(old_state, new_df, "Sütun Silme", desc_msg)
                        st.session_state[StateKeys.ACTIVE_DF] = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                            "action_name": "Sütun Yönetimi",
                            "description": desc_msg,
                            "duration_ms": duration
                        }
                        NotificationHub.notify_success(desc_msg)
                        st.rerun()

            with col_manage_tab2:
                col_to_cast = st.selectbox("Dönüştürülecek Sütun:", df.columns, key="cast_col_select")
                target_type = st.selectbox("Hedef Tip:", ["Sayısal (Numeric / Float)", "Tamsayı (Integer)", "Tarih / Saat (Datetime)", "Metin / Kategori (String)"])

                if st.button("🔄 Veri Tipini Dönüştür", width="stretch"):
                    val_rep, succ_rate = InputValidator.validate_type_cast(df[col_to_cast], target_type)
                    if not val_rep.is_valid:
                        for iss in val_rep.get_errors():
                            st.error(f"❌ {iss.message}")
                        st.stop()

                    start_t = time.time()
                    with st.spinner("Veri tipi dönüştürülüyor..."):
                        old_state = df.copy()
                        new_df, desc_msg = apply_type_cast(df, col_to_cast, target_type)
                        history_mgr.record_change(old_state, new_df, "Tip Dönüştürme", desc_msg)
                        st.session_state[StateKeys.ACTIVE_DF] = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                            "action_name": "Tip Dönüştürme",
                            "description": desc_msg,
                            "duration_ms": duration
                        }
                        NotificationHub.notify_success(desc_msg)
                        st.rerun()

    st.markdown("---")
    if st.button("🔄 TÜM DEĞİŞİKLİKLERİ SIFIRLA VE İLK HALİNE DÖN", type="secondary", width="stretch"):
        with st.spinner("Veri seti ilk haline döndürülüyor..."):
            st.session_state[StateKeys.ACTIVE_DF] = raw_df.copy()
            st.session_state[StateKeys.HISTORY_MGR] = DataHistoryManager(max_history=15)
            clear_automl_result()
            st.session_state[StateKeys.LAST_OP_FEEDBACK] = {
                "action_name": "Sıfırlama",
                "description": "Veri seti orijinal haline geri döndürüldü.",
                "duration_ms": 10.0
            }
            NotificationHub.notify_info("Veri seti orijinal haline döndürüldü.")
            st.rerun()
