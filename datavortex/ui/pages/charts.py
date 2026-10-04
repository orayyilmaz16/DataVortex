"""
DataVortex — Sayfa 4: Gelişmiş İnteraktif Görselleştirme Stüdyosu (Chart Builder)
"""

import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_charts_page(df: pd.DataFrame, theme_plot: str) -> None:
    """11 farklı Plotly grafik türünü yapılandırma ve dışa aktarma arayüzünü render eder."""
    st.markdown("### 📈 Gelişmiş İnteraktif Görselleştirme Stüdyosu")

    col_chart_cfg, col_chart_view = st.columns([1, 3])

    with col_chart_cfg:
        st.markdown("#### ⚙️ Grafik Ayarları")
        chart_type = st.selectbox(
            "Grafik Türü",
            [
                "📊 Bar / Çubuk Grafiği",
                "📉 Çizgi (Line / Trend) Grafiği",
                "🔵 Saçılım (Scatter / Dağılım)",
                "📦 Kutu (Box Plot) Grafiği",
                "🎻 Keman (Violin Plot) Grafiği",
                "📊 Histogram & Yoğunluk",
                "🥧 Pasta / Halka (Donut) Grafiği",
                "🗺️ Ağaç Haritası (Treemap)",
                "☀️ Güneş Patlaması (Sunburst)",
                "🧊 3D Saçılım (3D Scatter)",
                "🌡️ 2D Yoğunluk Isı Haritası"
            ]
        )

        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        x_axis = st.selectbox("X Ekseni", all_cols, index=0)
        y_default = num_cols[0] if num_cols else all_cols[0]
        y_axis = st.selectbox("Y Ekseni", all_cols, index=all_cols.index(y_default) if y_default in all_cols else 0)

        color_col = st.selectbox("Renk / Gruplama (Color)", ["Yok"] + all_cols, index=0)
        color_val = None if color_col == "Yok" else color_col

        size_col = None
        if chart_type in ["🔵 Saçılım (Scatter / Dağılım)", "🧊 3D Saçılım (3D Scatter)"]:
            size_choice = st.selectbox("Baloncuk Boyutu (Size)", ["Yok"] + num_cols)
            size_col = None if size_choice == "Yok" else size_choice

        z_axis = None
        if chart_type == "🧊 3D Saçılım (3D Scatter)":
            z_axis = st.selectbox("Z Ekseni", num_cols if num_cols else all_cols)

        chart_title = st.text_input("Grafik Başlığı", value=f"{chart_type.split(' ')[1]} Analizi")
        chart_height = st.slider("Grafik Yüksekliği (px)", min_value=350, max_value=800, value=500, step=50)

    with col_chart_view:
        try:
            fig = None
            plot_df = df.copy()

            # NaN ve negatif boyut koruması
            safe_size_col = None
            if size_col and size_col in plot_df.columns:
                s_size = pd.to_numeric(plot_df[size_col], errors="coerce")
                if s_size.notna().sum() > 0:
                    fill_val = s_size.median() if (s_size.median() is not None and s_size.median() > 0) else (s_size.mean() if s_size.mean() > 0 else 1.0)
                    s_size = s_size.fillna(fill_val).abs()
                    if (s_size <= 0).all():
                        s_size = s_size + 1.0
                    plot_df[f"_safe_size_{size_col}"] = s_size
                    safe_size_col = f"_safe_size_{size_col}"
                else:
                    st.info(f"💡 '{size_col}' sütunundaki tüm değerler eksik/geçersiz olduğundan baloncuk boyutu varsayılana ayarlandı.")

            # Değer sütunları (pie, treemap, sunburst) için NaN ve negatif koruması
            safe_val_col = None
            if y_axis in num_cols:
                s_val = pd.to_numeric(plot_df[y_axis], errors="coerce").fillna(0)
                if chart_type in ["🥧 Pasta / Halka (Donut) Grafiği", "🗺️ Ağaç Haritası (Treemap)", "☀️ Güneş Patlaması (Sunburst)"]:
                    s_val = s_val.clip(lower=0)
                plot_df[f"_safe_val_{y_axis}"] = s_val
                safe_val_col = f"_safe_val_{y_axis}"

            if chart_type == "📊 Bar / Çubuk Grafiği":
                fig = px.bar(plot_df, x=x_axis, y=y_axis, color=color_val, title=chart_title, template=theme_plot, barmode="group")
            elif chart_type == "📉 Çizgi (Line / Trend) Grafiği":
                fig = px.line(plot_df, x=x_axis, y=y_axis, color=color_val, title=chart_title, template=theme_plot, markers=True)
            elif chart_type == "🔵 Saçılım (Scatter / Dağılım)":
                trend_opt = None
                if x_axis in num_cols and y_axis in num_cols:
                    valid_pairs = plot_df[[x_axis, y_axis]].dropna()
                    if len(valid_pairs) > 3 and valid_pairs[x_axis].nunique() > 1 and valid_pairs[y_axis].nunique() > 1:
                        trend_opt = "ols"
                fig = px.scatter(
                    plot_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_val,
                    size=safe_size_col,
                    trendline=trend_opt,
                    title=chart_title,
                    template=theme_plot
                )
            elif chart_type == "📦 Kutu (Box Plot) Grafiği":
                fig = px.box(plot_df, x=x_axis, y=y_axis, color=color_val, title=chart_title, template=theme_plot, points="outliers")
            elif chart_type == "🎻 Keman (Violin Plot) Grafiği":
                fig = px.violin(plot_df, x=x_axis, y=y_axis, color=color_val, box=True, points="all", title=chart_title, template=theme_plot)
            elif chart_type == "📊 Histogram & Yoğunluk":
                fig = px.histogram(plot_df, x=x_axis, y=y_axis if y_axis != x_axis else None, color=color_val, marginal="box", title=chart_title, template=theme_plot)
            elif chart_type == "🥧 Pasta / Halka (Donut) Grafiği":
                fig = px.pie(plot_df, names=x_axis, values=safe_val_col, hole=0.4, title=chart_title, template=theme_plot)
            elif chart_type == "🗺️ Ağaç Haritası (Treemap)":
                path_cols = [x_axis]
                if color_val and color_val != x_axis:
                    path_cols.append(color_val)
                fig = px.treemap(plot_df, path=path_cols, values=safe_val_col, title=chart_title, template=theme_plot)
            elif chart_type == "☀️ Güneş Patlaması (Sunburst)":
                path_cols = [x_axis]
                if color_val and color_val != x_axis:
                    path_cols.append(color_val)
                fig = px.sunburst(plot_df, path=path_cols, values=safe_val_col, title=chart_title, template=theme_plot)
            elif chart_type == "🧊 3D Saçılım (3D Scatter)":
                fig = px.scatter_3d(
                    plot_df,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color=color_val,
                    size=safe_size_col,
                    title=chart_title,
                    template=theme_plot
                )
            elif chart_type == "🌡️ 2D Yoğunluk Isı Haritası":
                fig = px.density_heatmap(plot_df, x=x_axis, y=y_axis, title=chart_title, template=theme_plot, marginal_x="histogram", marginal_y="histogram")

            if fig is not None:
                fig.update_layout(height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width="stretch")

                html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                st.download_button(
                    label="📥 Bu Grafiği İnteraktif HTML Olarak İndir",
                    data=html_bytes,
                    file_name=f"datavortex_chart_{int(time.time())}.html",
                    mime="text/html",
                    width="stretch"
                )
        except Exception as e:
            st.error(f"Grafik oluşturulamadı: {e}")
