"""
================================================================================
⚡ DATAVORTEX — AKILLI EXCEL VERİ BİLİMİ, ANALİTİK & AUTOML PLATFORMU
================================================================================
Modern, zengin özellikli, tam döngülü Veri Bilimi, İstatistik, Keşifçi Veri
Analizi (EDA), 1-Tık Otomatik Makine Öğrenmesi (AutoML), Model Açıklanabilirliği,
İnteraktif Tahmin Simülatörü, Doğrulama Mimarisi ve Sistem Test Merkezi.
================================================================================
"""

import warnings
from typing import Any, Dict, Optional
import streamlit as st

# Scikit-Learn ve kütüphane uyarılarını temiz tut
warnings.filterwarnings("ignore")

from datavortex.core.config import APP_TITLE, APP_ICON, APP_LAYOUT
from datavortex.core.state import (
    StateKeys,
    initialize_session_state,
    update_active_dataset_if_source_changed,
)
from datavortex.data.history import DataHistoryManager
from datavortex.validation.schema import SchemaValidator
from datavortex.ui.components import UIComponents
from datavortex.ui.sidebar import render_sidebar
from datavortex.ui.pages import (
    render_explorer_page,
    render_cleaning_page,
    render_eda_page,
    render_charts_page,
    render_ml_studio_page,
    render_insights_page,
    render_export_page,
)


# ==============================================================================
# 1. SAYFA KONFİGÜRASYONU & ÖZEL MODERN CSS
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

UIComponents.apply_custom_css()


initialize_session_state(DataHistoryManager)

raw_df, file_meta, uploaded_file, selected_sheet, header_row, theme_plot, data_source = render_sidebar()


# ==============================================================================
# 5. BAŞLIK & HERO BANNER
# ==============================================================================

UIComponents.render_hero_banner()


# Veri yoksa karşılama vitrini göster
if raw_df is None or raw_df.empty:
    st.info("👈 Analize başlamak için lütfen sol taraftaki menüden bir Excel/CSV dosyası yükleyin veya **'Hazır Örnek Veri Seti Kullan'** seçeneğini işaretleyin.")

    st.markdown("### 🌟 Platform Yetenekleri")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h4>🤖 1-Tık AutoML Motoru</h4>
            <p style='color:#94A3B8; font-size:0.9rem;'>Sınıflandırma ve Regresyon modellerini tek tıkla otomatik eğitir, kıyaslar ve en yüksek performanslı şampiyon modeli seçer.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h4>🔮 Canlı Tahmin & Açıklanabilirlik</h4>
            <p style='color:#94A3B8; font-size:0.9rem;'>Özellik önem düzeyleri, Confusion Matrix, ROC eğrileri ve anlık kullanıcı girdileriyle çalışan interaktif simülatör.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="glass-card">
            <h4>🛡️ Validasyon & Güvenlik Mimarisi</h4>
            <p style='color:#94A3B8; font-size:0.9rem;'>AST tabanlı güvenli formül motoru, XSS koruması, ML muhafızı ve veri kalitesi puanlama motoru.</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ==============================================================================
# 6. AKTİF VERİ ÇERÇEVESİ YÖNETİMİ
# ==============================================================================

current_source_sig = (
    data_source,
    uploaded_file.name if uploaded_file is not None else "sample",
    selected_sheet,
    header_row,
    file_meta.get("encoding", ""),
    file_meta.get("delimiter", "")
)

update_active_dataset_if_source_changed(raw_df, current_source_sig, DataHistoryManager)

df = st.session_state.active_df
schema_report = SchemaValidator.validate_dataframe(df)


# ==============================================================================
# 6. ÜST ÖZET KPI KARTLARI, DOSYA BİLGİSİ & İŞLEM BİLDİRİMİ
# ==============================================================================

if uploaded_file is not None and file_meta:
    UIComponents.render_file_upload_card(uploaded_file.name, file_meta)

if st.session_state.last_op_feedback:
    fb = st.session_state.last_op_feedback
    UIComponents.render_operation_feedback(
        action_name=fb["action_name"],
        description=fb["description"],
        affected_rows=fb.get("affected_rows", 0),
        duration_ms=fb.get("duration_ms", 0.0)
    )

health_dict = {
    "score": schema_report.score,
    "grade": schema_report.grade,
    "missing_cells": schema_report.metrics.get("missing_cells", 0),
    "missing_pct": schema_report.metrics.get("missing_pct", 0.0),
    "dup_rows": schema_report.metrics.get("dup_rows", 0),
    "dup_pct": schema_report.metrics.get("dup_pct", 0.0),
    "color": "#22C55E" if schema_report.score >= 80 else ("#F59E0B" if schema_report.score >= 60 else "#EF4444")
}

UIComponents.render_kpi_cards(df, health_dict)


# ==============================================================================
# 7. ANA SEKMELİ VERİ BİLİMİ & AUTOML MİMARİSİ
# ==============================================================================

tab_explorer, tab_cleaning, tab_eda, tab_charts, tab_stats_ml, tab_insights, tab_export = st.tabs([
    "📁 1. Veri Gezgini & Şema",
    "🧹 2. Veri Temizleme & Manipülasyon",
    "📊 3. Keşifçi Analiz (EDA)",
    "📈 4. İnteraktif Grafikler",
    "🤖 5. 1-Tık AutoML & ML Stüdyosu",
    "💡 6. Akıllı İçgörüler & AI",
    "💾 7. Dışa Aktar & Rapor"
])


# ==============================================================================
# SEKME 1: VERİ GEZGİNİ & ŞEMA DENETÇİSİ
# ==============================================================================
with tab_explorer:
    render_explorer_page(df)


# ==============================================================================
# SEKME 2: VERİ TEMİZLEME, MANİPÜLASYON & GERİ ALMA (UNDO/REDO)
# ==============================================================================
with tab_cleaning:
    render_cleaning_page(df, raw_df)


# ==============================================================================
# SEKME 3: KEŞİFÇİ VERİ ANALİZİ (EDA)
# ==============================================================================
with tab_eda:
    render_eda_page(df, theme_plot)


# ==============================================================================
# SEKME 4: GELİŞMİŞ İNTERAKTİF GÖRSELLEŞTİRME STÜDYOSU
# ==============================================================================
with tab_charts:
    render_charts_page(df, theme_plot)


# ==============================================================================
# SEKME 5: 1-TIK AUTOML & MAKİNE ÖĞRENMESİ STÜDYOSU
# ==============================================================================
with tab_stats_ml:
    render_ml_studio_page(df, theme_plot)


# ==============================================================================
# SEKME 6: AKILLI İÇGÖRÜLER & AI
# ==============================================================================
with tab_insights:
    render_insights_page(df, schema_report, health_dict)


# ==============================================================================
# SEKME 7: DIŞA AKTARMA & RAPORLAMA
# ==============================================================================
with tab_export:
    render_export_page(df, schema_report)


# ==============================================================================
# ALT BİLGİ (FOOTER)
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 15px;'>"
    "⚡ DataVortex Akıllı Excel Veri Bilimi & AutoML Platformu • Streamlit, Plotly, Scikit-Learn, SciPy & Google Gemini AI ile Güçlendirildi."
    "</div>",
    unsafe_allow_html=True
)