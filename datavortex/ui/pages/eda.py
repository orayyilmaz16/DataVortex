"""
DataVortex — Sayfa 3: Kapsamlı Keşifçi Veri Analizi (EDA)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_eda_page(df: pd.DataFrame, theme_plot: str) -> None:
    """Keşifçi veri analizi (sayısal, kategorik ve korelasyon) arayüzünü render eder."""
    st.markdown("### 📊 Kapsamlı Keşifçi Veri Analizi (EDA)")

    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    eda_sub_1, eda_sub_2, eda_sub_3 = st.tabs([
        "📈 Sayısal İstatistikler",
        "🏷️ Kategorik Değişken Özeti",
        "🔥 Korelasyon & İlişki Matrisi"
    ])

    with eda_sub_1:
        if not num_df.empty:
            st.markdown("#### 📐 Sayısal Değişkenlerin Detaylı Dağılım Metrikleri")
            desc = num_df.describe().T
            desc["IQR"] = desc["75%"] - desc["25%"]
            desc["Skewness (Çarpıklık)"] = num_df.skew()
            desc["Kurtosis (Basıklık)"] = num_df.kurtosis()
            desc["Variance (Varyans)"] = num_df.var()
            st.dataframe(desc.style.format("{:.2f}"), width="stretch")

            st.markdown("#### 📦 Sayısal Değişken Dağılım İncelemesi")
            selected_num_col = st.selectbox("İncelenecek Sayısal Sütun:", num_df.columns, key="eda_num_col")
            c_eda1, c_eda2 = st.columns(2)
            with c_eda1:
                fig_hist = px.histogram(
                    df, x=selected_num_col, marginal="box",
                    title=f"{selected_num_col} - Histogram ve Kutu Dağılımı",
                    template=theme_plot, color_discrete_sequence=["#6366F1"]
                )
                st.plotly_chart(fig_hist, width="stretch")
            with c_eda2:
                fig_ecdf = px.ecdf(
                    df, x=selected_num_col,
                    title=f"{selected_num_col} - Kümülatif Dağılım (ECDF)",
                    template=theme_plot, color_discrete_sequence=["#EC4899"]
                )
                st.plotly_chart(fig_ecdf, width="stretch")
        else:
            st.warning("Veri setinde incelenecek sayısal sütun bulunmuyor.")

    with eda_sub_2:
        if not cat_df.empty:
            st.markdown("#### 🏷️ Kategorik Sütun Frekans ve Dağılımı")
            selected_cat_col = st.selectbox("İncelenecek Kategorik Sütun:", cat_df.columns, key="eda_cat_col")

            top_counts = df[selected_cat_col].value_counts(dropna=False).reset_index()
            top_counts.columns = [selected_cat_col, "Frekans"]
            top_counts["Oran %"] = round((top_counts["Frekans"] / max(len(df), 1)) * 100, 2)

            c_cat1, c_cat2 = st.columns([1, 2])
            with c_cat1:
                st.dataframe(top_counts, width="stretch", height=350)
            with c_cat2:
                fig_cat = px.bar(
                    top_counts.head(15), x=selected_cat_col, y="Frekans",
                    text="Frekans", color="Frekans",
                    title=f"{selected_cat_col} Sütunu Değer Dağılımı (İlk 15)",
                    template=theme_plot, color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_cat, width="stretch")
        else:
            st.info("Veri setinde incelenecek kategorik/metin sütunu bulunmuyor.")

    with eda_sub_3:
        if len(num_df.columns) >= 2:
            st.markdown("#### 🔥 Korelasyon Isı Haritası")
            corr_method = st.radio("Korelasyon Yöntemi", ["Pearson (Doğrusal)", "Spearman (Sıralı / Non-lineer)"], horizontal=True)
            method_key = "pearson" if corr_method.startswith("Pearson") else "spearman"

            corr_matrix = num_df.corr(method=method_key, numeric_only=True)

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"{corr_method} Korelasyon Isı Haritası",
                template=theme_plot
            )
            st.plotly_chart(fig_corr, width="stretch")

            st.markdown("#### ⚡ En Güçlü Korelasyona Sahip Değişken Çiftleri")
            corr_unstack = corr_matrix.unstack().reset_index()
            corr_unstack.columns = ["Değişken 1", "Değişken 2", "Korelasyon"]
            corr_unstack = corr_unstack[corr_unstack["Değişken 1"] < corr_unstack["Değişken 2"]]
            corr_unstack["Mutlak Güç"] = corr_unstack["Korelasyon"].abs()
            top_corrs = corr_unstack.sort_values(by="Mutlak Güç", ascending=False).head(10)
            st.dataframe(top_corrs[["Değişken 1", "Değişken 2", "Korelasyon"]].style.format({"Korelasyon": "{:.3f}"}), width="stretch")
        else:
            st.info("Korelasyon analizi için en az 2 sayısal sütun gereklidir.")
