"""
DataVortex — Sayfa 5: 1-Tık AutoML & İstatistik Stüdyosu (ML Studio)
"""

import io
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datavortex.analytics.statistics import StatsEngine
from datavortex.core.state import (
    StateKeys,
    get_automl_result,
    set_automl_result,
)
from datavortex.data.io import load_file
from datavortex.ml.automl import get_recommended_target_index, train_models
from datavortex.ml.explainability import (
    get_classification_confusion_matrix,
    get_model_feature_importances,
    get_regression_diagnostics,
)
from datavortex.ml.prediction import predict_batch, predict_single
from datavortex.ui.components import NotificationHub, UIComponents
from datavortex.validation.ml_guard import MLGuard


def render_ml_studio_page(df: pd.DataFrame, theme_plot: str) -> None:
    """AutoML, model açıklanabilirliği, tahmin ve istatistik stüdyosunu render eder."""
    st.markdown("### 🤖 1-Tık Akıllı Makine Öğrenmesi (AutoML) & İstatistik Stüdyosu")

    ml_sub_1, ml_sub_2, ml_sub_3, ml_sub_4, ml_sub_5, ml_sub_6 = st.tabs([
        "🚀 1. 1-Tık AutoML & Model Liderlik Tablosu",
        "🔮 2. Canlı Tahmin Simülatörü & Toplu Tahmin",
        "🎯 3. K-Means Kümeleme",
        "🚨 4. Anomali & Aykırı Değer Tespiti (Isolation Forest)",
        "🧩 5. Boyut İndirgeme (PCA)",
        "🧪 6. İstatistiksel Hipotez Testleri"
    ])

    # 5.1 1-TIK AUTOML
    with ml_sub_1:
        st.markdown("#### 🚀 Otomatik Model Eğitimi, Kıyaslama & Şampiyon Model Seçimi")
        all_columns = df.columns.tolist()
        rec_target_idx = get_recommended_target_index(all_columns, df)

        c_auto1, c_auto2, c_auto3 = st.columns([1.5, 1.5, 1])
        with c_auto1:
            target_col = st.selectbox("🎯 Hedef Değişken (Target / Y):", all_columns, index=rec_target_idx)

        target_series = df[target_col].dropna()
        n_unique = target_series.nunique()
        is_num = pd.api.types.is_numeric_dtype(target_series)
        is_date = pd.api.types.is_datetime64_any_dtype(target_series) or "tarih" in target_col.lower() or "date" in target_col.lower()

        if is_date:
            detected_task = "⚠️ Uygun Olmayan Hedef (Tarih / Zaman)"
        elif not is_num or (is_num and n_unique <= 10):
            detected_task = "Sınıflandırma (Classification)"
        else:
            detected_task = "Regresyon (Regression)"

        with c_auto2:
            task_options = ["Sınıflandırma (Classification)", "Regresyon (Regression)"]
            default_task_idx = 0 if detected_task.startswith("Sınıflandırma") else (1 if detected_task.startswith("Regresyon") else 0)
            task_type = st.selectbox("🧠 Makine Öğrenmesi Görevi:", task_options, index=default_task_idx)

        with c_auto3:
            test_size = st.slider("🧪 Test Veri Oranı (%):", min_value=10, max_value=40, value=20, step=5) / 100.0

        candidate_features = [c for c in all_columns if c != target_col]
        selected_features = st.multiselect("🧩 Modele Girecek Bağımsız Değişkenler (Features / X):", candidate_features, default=candidate_features)

        if st.button("🚀 1-Tıkla Tüm Modelleri Eğit ve Karşılaştır", type="primary", width="stretch"):
            val_report = MLGuard.validate_ml_setup(df, target_col, task_type, selected_features)
            if not val_report.is_valid:
                for err in val_report.get_errors():
                    st.error(f"❌ {err.message}")
                st.stop()

            for warn in val_report.get_warnings():
                st.warning(f"⚠️ {warn.message}")

            progress_bar = st.progress(0, text="Modeller eğitiliyor...")

            def progress_update(ratio: float, text: str) -> None:
                progress_bar.progress(ratio, text=text)

            try:
                start_t = time.time()
                result = train_models(
                    df=df,
                    target_col=target_col,
                    task_type=task_type,
                    selected_features=selected_features,
                    test_size=test_size,
                    progress_cb=progress_update
                )
                progress_bar.empty()
                set_automl_result(result)
                NotificationHub.notify_success(f"AutoML tamamlandı! Şampiyon Model: {result['best_model_name']}")
            except Exception as ex:
                progress_bar.empty()
                st.error(f"❌ Eğitim sırasında hata oluştu: {str(ex)}")

        # Sonuçları Göster
        automl_res = get_automl_result()
        if automl_res is not None:
            lead_df = automl_res["leaderboard_df"].copy()
            best_m_name = automl_res["best_model_name"]
            is_classif = "Sınıflandırma" in automl_res["task_type"]

            st.markdown("---")
            primary_metric_label = "F1-Score" if is_classif else "R² Skoru"
            primary_metric_val = f"%{lead_df.iloc[0]['F1-Score (%)']}" if is_classif else f"{lead_df.iloc[0]['R² Skoru']}"

            UIComponents.render_champion_card(
                best_model_name=best_m_name,
                target_col=automl_res["target_col"],
                task_type=automl_res["task_type"],
                primary_metric_label=primary_metric_label,
                primary_metric_val=primary_metric_val,
                duration_ms=lead_df.iloc[0]["Eğitim Süresi (ms)"]
            )

            st.markdown("#### 📊 Model Liderlik Tablosu (Leaderboard)")
            display_lead = lead_df.drop(columns=["_score_sort"])
            st.dataframe(
                display_lead.style.highlight_max(
                    subset=["F1-Score (%)", "Accuracy (%)"] if is_classif else ["R² Skoru"],
                    color="rgba(34, 197, 94, 0.3)"
                ),
                width="stretch"
            )

            st.markdown("#### 🔍 Şampiyon Model Detaylı Analizi & Açıklanabilirlik")
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                imp_df, imp_err = get_model_feature_importances(automl_res)
                if imp_df is not None:
                    fig_imp = px.bar(
                        imp_df, x="Önem Düzeyi", y="Özellik", orientation="h",
                        title=f"{best_m_name} - En Önemli Değişkenler (Feature Importance)",
                        template=theme_plot, color="Önem Düzeyi", color_continuous_scale="Tealgrn"
                    )
                    st.plotly_chart(fig_imp, width="stretch")
                elif imp_err:
                    st.caption(f"Özellik önemi grafiği: {imp_err}")

            with col_viz2:
                if is_classif:
                    cm, labels, _ = get_classification_confusion_matrix(automl_res)
                    fig_cm = px.imshow(
                        cm,
                        x=[f"Tahmin: {l}" for l in labels],
                        y=[f"Gerçek: {l}" for l in labels],
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title=f"{best_m_name} - Karmaşıklık Matrisi (Confusion Matrix)",
                        template=theme_plot
                    )
                    st.plotly_chart(fig_cm, width="stretch")
                else:
                    y_test, y_test_pred, min_v, max_v = get_regression_diagnostics(automl_res)
                    fig_reg = px.scatter(
                        x=y_test, y=y_test_pred,
                        labels={"x": "Gerçek Değerler", "y": "Tahmin Edilen Değerler"},
                        title=f"{best_m_name} - Gerçek vs. Tahmin Regresyon Grafiği",
                        template=theme_plot
                    )
                    fig_reg.add_trace(go.Scatter(
                        x=[min_v, max_v], y=[min_v, max_v],
                        mode="lines", name="İdeal Tahmin (Y=X)",
                        line=dict(color="#EF4444", dash="dash")
                    ))
                    st.plotly_chart(fig_reg, width="stretch")

    # 5.2 CANLI TAHMİN SİMÜLATÖRÜ & TOPLU TAHMİN
    with ml_sub_2:
        st.markdown("#### 🔮 İnteraktif 'What-If' Tahmin Simülatörü & Toplu Tahmin")
        automl_res = get_automl_result()
        if automl_res is None:
            st.info("👈 Tahmin yapabilmek için lütfen önce **'1. 1-Tık AutoML & Model Liderlik Tablosu'** sekmesinden modellerinizi eğitin.")
        else:
            best_pipe = automl_res["best_pipe"]
            best_m_name = automl_res["best_model_name"]
            selected_features = automl_res["selected_features"]
            is_classif = "Sınıflandırma" in automl_res["task_type"]

            sim_tab1, sim_tab2 = st.tabs(["⚡ Tekil Canlı Tahmin (What-If)", "📁 Toplu Tahmin (Batch Prediction)"])

            with sim_tab1:
                st.markdown(f"**Aktif Model:** `{best_m_name}` | **Hedef Değişken:** `{automl_res['target_col']}`")
                raw_inputs = {}
                num_cols_layout = min(3, len(selected_features))
                cols_grid = st.columns(num_cols_layout)

                for i, feat in enumerate(selected_features):
                    col_target = cols_grid[i % num_cols_layout]
                    series = df[feat].dropna()
                    with col_target:
                        is_feat_dt = pd.api.types.is_datetime64_any_dtype(series) or "tarih" in feat.lower() or "date" in feat.lower()
                        if is_feat_dt:
                            raw_inputs[feat] = st.date_input(f"📅 {feat}", value=pd.to_datetime("today"))
                        elif pd.api.types.is_numeric_dtype(series):
                            min_v = float(series.min()) if not series.empty else 0.0
                            max_v = float(series.max()) if not series.empty else 100.0
                            mean_v = float(series.mean()) if not series.empty else 50.0
                            step_v = (max_v - min_v) / 100.0 if max_v != min_v else 1.0
                            raw_inputs[feat] = st.number_input(f"🔢 {feat}", value=round(mean_v, 2), step=round(step_v, 2))
                        else:
                            unique_opts = series.unique().tolist()
                            if not unique_opts:
                                unique_opts = ["Bilinmiyor"]
                            raw_inputs[feat] = st.selectbox(f"🏷️ {feat}", options=unique_opts, index=0)

                if st.button("⚡ Hemen Tahmin Et", type="primary", width="stretch"):
                    with st.spinner("Tahmin hesaplanıyor..."):
                        pred_val, prob_df, err = predict_single(
                            best_pipe=best_pipe,
                            raw_inputs=raw_inputs,
                            engineered_features=automl_res["engineered_features"],
                            is_classification=is_classif
                        )
                        if err:
                            st.error(f"❌ {err}")
                        else:
                            st.markdown("### 🎯 Tahmin Sonucu")
                            if is_classif:
                                st.success(f"📌 Modelin Sınıf Tahmini: **{pred_val}**")
                                if prob_df is not None:
                                    fig_prob = px.bar(
                                        prob_df, x="Sınıf", y="Olasılık (%)", text="Olasılık (%)",
                                        title="Sınıf Olasılık Dağılımı (Confidence)",
                                        template=theme_plot, color="Olasılık (%)", color_continuous_scale="Blues"
                                    )
                                    st.plotly_chart(fig_prob, width="stretch")
                            else:
                                st.success(f"📌 Tahmin Edilen Değer: **{pred_val:,.2f}**")
                            NotificationHub.notify_success("Tahmin başarıyla üretildi.")

            with sim_tab2:
                st.markdown("#### 📁 Yeni Dosya ile Toplu Tahmin Yapın")
                batch_file = st.file_uploader("Tahmin Yapılacak Excel/CSV Dosyası", type=["xlsx", "xls", "xlsm", "csv", "tsv", "txt"], key="batch_uploader")
                target_batch_df = None

                if batch_file is not None:
                    try:
                        b_bytes = batch_file.getvalue()
                        target_batch_df, _ = load_file(b_bytes, batch_file.name)
                        st.caption(f"Yüklenen dosya: **{len(target_batch_df):,}** satır × **{len(target_batch_df.columns)}** sütun.")
                    except Exception as b_err:
                        st.error(f"Dosya okunamadı: {b_err}")
                else:
                    if st.button("🧪 Mevcut Test Verisi ile Toplu Tahmin Çalıştır"):
                        target_batch_df = df.loc[automl_res["X_test"].index, selected_features].copy()

                if target_batch_df is not None:
                    with st.spinner("Toplu tahminler hesaplanıyor..."):
                        out_batch_df, b_err = predict_batch(
                            best_pipe=best_pipe,
                            target_batch_df=target_batch_df,
                            selected_features=selected_features,
                            engineered_features=automl_res["engineered_features"]
                        )
                        if b_err:
                            st.error(f"❌ {b_err}")
                        else:
                            st.markdown("##### 📋 Tahmin Sonuçları Önizleme")
                            st.dataframe(out_batch_df.head(20), width="stretch")
                            c_d1, c_d2 = st.columns(2)
                            with c_d1:
                                exc_buffer = io.BytesIO()
                                with pd.ExcelWriter(exc_buffer, engine="openpyxl") as writer:
                                    out_batch_df.to_excel(writer, index=False, sheet_name="Tahminler")
                                st.download_button(
                                    label="📥 Tahminleri Excel (.xlsx) Olarak İndir",
                                    data=exc_buffer.getvalue(),
                                    file_name=f"model_tahminleri_{int(time.time())}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    width="stretch"
                                )
                            with c_d2:
                                csv_b_data = out_batch_df.to_csv(index=False).encode("utf-8-sig")
                                st.download_button(
                                    label="📄 Tahminleri CSV (.csv) Olarak İndir",
                                    data=csv_b_data,
                                    file_name=f"model_tahminleri_{int(time.time())}.csv",
                                    mime="text/csv",
                                    width="stretch"
                                )
                            NotificationHub.notify_success("Toplu tahmin dosyası hazırlandı.")

    # 5.3 K-MEANS KÜMELEME
    with ml_sub_3:
        st.markdown("#### 🎯 K-Means Kümeleme & Müşteri / Veri Segmentasyonu")
        num_cols_km = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols_km) >= 2:
            cluster_features = st.multiselect("Kümelemede Kullanılacak Sayısal Değişkenler:", num_cols_km, default=num_cols_km[:min(4, len(num_cols_km))], key="km_feats")
            if len(cluster_features) >= 2:
                col_k1, col_k2 = st.columns([1, 2])
                with col_k1:
                    selected_k = st.slider("Uygulanacak Küme Sayısı (K):", min_value=2, max_value=min(8, max(2, len(df) - 1)), value=3, key="km_slider")

                with st.spinner("K-Means kümeleme çalıştırılıyor..."):
                    km_res = StatsEngine.run_kmeans_clustering(df, cluster_features, k=selected_k)
                    if "error" in km_res:
                        st.error(km_res["error"])
                    else:
                        with col_k1:
                            fig_elbow = px.line(
                                x=km_res["k_range"], y=km_res["inertias"],
                                markers=True, title="Dirsek Yöntemi (Optimum K Seçimi)",
                                labels={"x": "Küme Sayısı (K)", "y": "Atalet (Inertia)"},
                                template=theme_plot
                            )
                            st.plotly_chart(fig_elbow, width="stretch")

                        with col_k2:
                            c_df_res = km_res["result_df"]
                            if len(cluster_features) >= 3:
                                fig_clust = px.scatter_3d(
                                    c_df_res, x=cluster_features[0], y=cluster_features[1], z=cluster_features[2],
                                    color="Küme", title=f"{selected_k} Kümeli 3D Görselleştirme",
                                    template=theme_plot
                                )
                            else:
                                fig_clust = px.scatter(
                                    c_df_res, x=cluster_features[0], y=cluster_features[1],
                                    color="Küme", title=f"{selected_k} Kümeli 2D Görselleştirme",
                                    template=theme_plot
                                )
                            st.plotly_chart(fig_clust, width="stretch")

                        st.markdown("##### 👥 Küme Profilleri (Ortalama Değerler)")
                        st.dataframe(km_res["cluster_profiles"].style.format(precision=2), width="stretch")
        else:
            st.info("Kümeleme için en az 2 sayısal değişken gereklidir.")

    # 5.4 ANOMALİ TESPİTİ (ISOLATION FOREST)
    with ml_sub_4:
        st.markdown("#### 🚨 1-Tık Anomali & Aykırı Değer Tespiti (Isolation Forest)")
        num_cols_anom = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols_anom) >= 2:
            anom_feats = st.multiselect("Anomali Analizinde Kullanılacak Sayısal Değişkenler:", num_cols_anom, default=num_cols_anom, key="anom_feats")
            c_anom1, c_anom2 = st.columns([1, 2])
            with c_anom1:
                contamination = st.slider("Beklenen Anomali Oranı (%):", min_value=1, max_value=15, value=5, step=1) / 100.0
                run_anom = st.button("🚨 Anomalileri Tespit Et", width="stretch")

            if run_anom and len(anom_feats) >= 2:
                with st.spinner("Isolation Forest anomalileri tespit ediyor..."):
                    an_res = StatsEngine.run_isolation_forest_anomaly(df, anom_feats, contamination=contamination)
                    if "error" in an_res:
                        st.error(an_res["error"])
                    else:
                        st.session_state[StateKeys.ANOMALY_RESULT] = an_res
                        NotificationHub.notify_success(f"{an_res['total_anomalies']} anomali tespit edildi.")

            if StateKeys.ANOMALY_RESULT in st.session_state:
                an_res = st.session_state[StateKeys.ANOMALY_RESULT]
                an_df = an_res["anom_df"]
                an_cnt = an_res["total_anomalies"]
                an_feats = an_res["features"]

                st.markdown(f"**Tespit Edilen Anomali Sayısı:** <span style='color:#EF4444; font-weight:700;'>{an_cnt} satır</span> (Toplamın %{an_res['anomaly_ratio']})", unsafe_allow_html=True)
                c_an_plot1, c_an_plot2 = st.columns(2)
                with c_an_plot1:
                    fig_anom = px.scatter(
                        an_df, x=an_feats[0], y=an_feats[1], color="Durum",
                        color_discrete_map={"🚨 Anomali (Aykırı)": "#EF4444", "✅ Normal": "#3B82F6"},
                        title="Anomali Dağılımı (İlk 2 Boyut)", template=theme_plot
                    )
                    st.plotly_chart(fig_anom, width="stretch")
                with c_an_plot2:
                    fig_score = px.histogram(
                        an_df, x="Anomali_Skoru", color="Durum",
                        color_discrete_map={"🚨 Anomali (Aykırı)": "#EF4444", "✅ Normal": "#3B82F6"},
                        title="Anomali Karar Skoru Dağılımı", template=theme_plot
                    )
                    st.plotly_chart(fig_score, width="stretch")

                st.markdown("##### 📋 Tespit Edilen Anomali Kayıtları")
                outliers_only = an_df[an_df["Durum"] == "🚨 Anomali (Aykırı)"].sort_values(by="Anomali_Skoru", ascending=True)
                st.dataframe(outliers_only, width="stretch")
        else:
            st.info("Anomali tespiti için en az 2 sayısal sütun gereklidir.")

    # 5.5 BOYUT İNDİRGEME (PCA)
    with ml_sub_5:
        st.markdown("#### 🧩 Temel Bileşen Analizi (PCA)")
        num_cols_pca = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_pca = df.select_dtypes(exclude=np.number).columns.tolist()
        if len(num_cols_pca) >= 3:
            pca_features = st.multiselect("PCA için Değişkenler:", num_cols_pca, default=num_cols_pca, key="pca_feats")
            if len(pca_features) >= 3:
                with st.spinner("PCA izdüşümü hesaplanıyor..."):
                    pca_res = StatsEngine.run_pca_decomposition(df, pca_features, n_components=3)
                    if "error" in pca_res:
                        st.error(pca_res["error"])
                    else:
                        st.write(f"**Açıklanan Toplam Varyans:** %{pca_res['total_explained_pct']} (PC1: %{pca_res['explained_variance_pct'][0]}, PC2: %{pca_res['explained_variance_pct'][1]})")
                        pca_df = pca_res["pca_df"]
                        if cat_cols_pca:
                            color_pca = st.selectbox("Grafik Renk Gruplayıcısı (Kategori):", ["Yok"] + cat_cols_pca, key="pca_color")
                            if color_pca != "Yok":
                                pca_df[color_pca] = df.loc[pca_df.index, color_pca].values
                        else:
                            color_pca = "Yok"

                        fig_pca = px.scatter(
                            pca_df, x="PC1", y="PC2",
                            color=None if color_pca == "Yok" else color_pca,
                            title="2D PCA İzdüşümü", template=theme_plot
                        )
                        st.plotly_chart(fig_pca, width="stretch")
        else:
            st.info("PCA için en az 3 sayısal sütun gereklidir.")

    # 5.6 İSTATİSTİKSEL HİPOTEZ TESTLERİ
    with ml_sub_6:
        st.markdown("#### 🧪 İstatistiksel Hipotez Testleri Merkezi")
        test_type = st.selectbox(
            "Uygulanacak Hipotez Testini Seçin:",
            [
                "Normallik Testi (Shapiro-Wilk / D'Agostino)",
                "İki Bağımsız Örneklem T-Testi",
                "Tek Yönlü ANOVA Testi (F-Testi)",
                "Ki-Kare Bağımsızlık Testi (Chi-Square)"
            ]
        )
        num_cols_ht = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_ht = df.select_dtypes(exclude=np.number).columns.tolist()

        if test_type.startswith("Normallik"):
            if num_cols_ht:
                norm_col = st.selectbox("Test Edilecek Sayısal Sütun:", num_cols_ht)
                norm_res = StatsEngine.test_normality(df[norm_col])
                if "error" in norm_res:
                    st.error(norm_res["error"])
                else:
                    st.write(f"**Test Adı:** {norm_res['test_name']}")
                    st.metric("Test İstatistiği", f"{norm_res['statistic']:.4f}")
                    st.metric("p-değeri (p-value)", f"{norm_res['p_value']:.4e}")
                    if norm_res["is_normal"]:
                        st.success(f"✅ **p = {norm_res['p_value']:.4f} > 0.05**: Dağılım %95 güven düzeyinde **NORMAL** dağılıma uygundur (H0 reddedilemez).")
                    else:
                        st.warning(f"⚠️ **p = {norm_res['p_value']:.4e} <= 0.05**: Dağılım **NORMAL DEĞİLDİR** (H0 reddedilir).")
            else:
                st.info("Sayısal sütun bulunamadı.")

        elif test_type.startswith("İki Bağımsız"):
            if num_cols_ht and cat_cols_ht:
                t_num = st.selectbox("Karşılaştırılacak Sayısal Değişken (Metrik):", num_cols_ht)
                t_cat = st.selectbox("Gruplayıcı Kategorik Değişken (2 gruplu):", cat_cols_ht)
                unique_groups = df[t_cat].dropna().unique()
                if len(unique_groups) >= 2:
                    g1 = st.selectbox("Grup 1:", unique_groups, index=0)
                    g2 = st.selectbox("Grup 2:", [g for g in unique_groups if g != g1], index=0)

                    s1 = df[df[t_cat] == g1][t_num].dropna()
                    s2 = df[df[t_cat] == g2][t_num].dropna()

                    t_res = StatsEngine.test_independent_ttest(s1, s2, label1=str(g1), label2=str(g2))
                    if "error" in t_res:
                        st.error(t_res["error"])
                    else:
                        st.metric("T-İstatistiği", f"{t_res['t_statistic']:.4f}")
                        st.metric("p-değeri", f"{t_res['p_value']:.4e}")
                        col_m1, col_m2 = st.columns(2)
                        col_m1.info(f"**{g1} Ortalaması:** {t_res['g1_mean']:.2f} (n={t_res['g1_n']})")
                        col_m2.info(f"**{g2} Ortalaması:** {t_res['g2_mean']:.2f} (n={t_res['g2_n']})")

                        if t_res["is_significant"]:
                            st.success(f"🎯 **Anlamlı Fark Var (p < 0.05)**: '{g1}' ve '{g2}' gruplarının {t_num} ortalamaları istatistiksel olarak anlamlı düzeyde farklıdır.")
                        else:
                            st.info("ℹ️ **Anlamlı Fark Yok (p >= 0.05)**: İki grup arasında istatistiksel olarak anlamlı bir fark tespit edilememiştir.")
            else:
                st.info("T-testi için hem sayısal hem kategorik sütun gereklidir.")

        elif test_type.startswith("Tek Yönlü ANOVA"):
            if num_cols_ht and cat_cols_ht:
                a_num = st.selectbox("Hedef Sayısal Değişken:", num_cols_ht)
                a_cat = st.selectbox("Gruplar:", cat_cols_ht)
                anova_res = StatsEngine.test_one_way_anova(df, a_num, a_cat)
                if "error" in anova_res:
                    st.error(anova_res["error"])
                else:
                    st.metric("F-İstatistiği", f"{anova_res['f_statistic']:.4f}")
                    st.metric("p-değeri", f"{anova_res['p_value']:.4e}")
                    if anova_res["is_significant"]:
                        st.success(f"🎯 **Anlamlı Fark Var (p < 0.05)**: Gruplardan en az birinin {a_num} ortalaması diğerlerinden anlamlı şekilde farklıdır.")
                    else:
                        st.info("ℹ️ Gruplar arasında istatistiksel olarak anlamlı bir ortalama farkı bulunamamıştır.")
                    st.dataframe(anova_res["group_stats_df"], width="stretch")
            else:
                st.info("ANOVA için hem sayısal hem kategorik değişken gereklidir.")

        elif test_type.startswith("Ki-Kare"):
            if len(cat_cols_ht) >= 2:
                c_col1 = st.selectbox("1. Kategorik Değişken:", cat_cols_ht, index=0)
                c_col2 = st.selectbox("2. Kategorik Değişken:", [c for c in cat_cols_ht if c != c_col1], index=0)
                chi_res = StatsEngine.test_chi_square(df, c_col1, c_col2)
                if "error" in chi_res:
                    st.error(chi_res["error"])
                else:
                    st.dataframe(chi_res["contingency_table"], width="stretch")
                    st.metric("Ki-Kare (Chi²) İstatistiği", f"{chi_res['chi2_statistic']:.4f}")
                    st.metric("p-değeri", f"{chi_res['p_value']:.4e}")
                    st.caption(f"Serbestlik Derecesi (df): {chi_res['dof']}")
                    if chi_res["is_significant"]:
                        st.success(f"🎯 **Bağımlı / İlişkili (p < 0.05)**: '{c_col1}' ve '{c_col2}' değişkenleri arasında anlamlı bir ilişki/bağımlılık vardır.")
                    else:
                        st.info(f"ℹ️ **Bağımsız (p >= 0.05)**: İki değişken birbirinden istatistiksel olarak bağımsızdır.")
            else:
                st.info("Ki-Kare testi için en az 2 kategorik değişken gereklidir.")
