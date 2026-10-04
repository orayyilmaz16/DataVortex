"""
DataVortex — Sayfa 1: Veri Gezgini ve Şema Denetçisi
"""

import pandas as pd
import streamlit as st


def render_explorer_page(df: pd.DataFrame) -> None:
    """Veri tablosunu ve detaylı şema özelliklerini interaktif olarak gösterir."""
    st.markdown("### 📋 Veri Tablosu ve Detaylı Şema İnceleme")

    col_exp_1, col_exp_2 = st.columns([3, 1])
    with col_exp_1:
        search_term = st.text_input("🔍 Sütunlarda Filtrele / Ara:", placeholder="Aramak istediğiniz anahtar kelime...")
    with col_exp_2:
        row_limit = st.selectbox("Gösterilecek Satır Sayısı", [50, 100, 250, 500, "Tümü"], index=0)

    filtered_df_display = df.copy()
    if search_term:
        mask = filtered_df_display.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_df_display = filtered_df_display[mask]
        st.caption(f"🔎 '{search_term}' araması için **{len(filtered_df_display)}** satır bulundu.")

    display_rows = len(filtered_df_display) if row_limit == "Tümü" else min(int(row_limit), len(filtered_df_display))
    st.dataframe(filtered_df_display.head(display_rows), width="stretch", height=420)

    st.markdown("#### 🧬 Sütun Özellikleri & Veri Tipleri")
    col_types_data = []
    for col in df.columns:
        s = df[col]
        non_null = s.count()
        null_cnt = s.isna().sum()
        null_ratio = round((null_cnt / max(len(df), 1)) * 100, 2)
        unique_cnt = s.nunique()
        sample_val = str(s.dropna().iloc[0]) if non_null > 0 else "Yok"
        col_types_data.append({
            "Sütun Adı": col,
            "Veri Tipi": str(s.dtype),
            "Dolu Değer": non_null,
            "Eksik Değer": null_cnt,
            "Eksik %": f"%{null_ratio}",
            "Tekil (Unique)": unique_cnt,
            "Örnek Değer": sample_val
        })
    st.dataframe(pd.DataFrame(col_types_data), width="stretch")
