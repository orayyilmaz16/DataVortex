"""
================================================================================
⚡ DATAVORTEX — AKILLI EXCEL VERİ BİLİMİ, ANALİTİK & AUTOML PLATFORMU
================================================================================
Modern, zengin özellikli, tam döngülü Veri Bilimi, İstatistik, Keşifçi Veri
Analizi (EDA), 1-Tık Otomatik Makine Öğrenmesi (AutoML), Model Açıklanabilirliği,
İnteraktif Tahmin Simülatörü, Doğrulama Mimarisi ve Sistem Test Merkezi.
================================================================================
"""

import io
import time
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

# Scikit-Learn ve kütüphane uyarılarını temiz tut
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

# Modüler DataVortex Kütüphanesi
from modules.validator import (
    ValidationSeverity,
    ValidationIssue,
    ValidationReport,
    SchemaValidator,
    InputValidator,
    FormulaValidator,
    MLGuard,
    DataStressTester,
)
from modules.data_processor import DataProcessor, DataHistoryManager
from modules.automl_engine import AutoMLEngine
from modules.stats_engine import StatsEngine
from modules.ui_components import UIComponents, NotificationHub



# ==============================================================================
# 1. SAYFA KONFİGÜRASYONU & ÖZEL MODERN CSS
# ==============================================================================

st.set_page_config(
    page_title="DataVortex | Akıllı Excel Veri Bilimi & AutoML Platformu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

UIComponents.apply_custom_css()


# ==============================================================================
# 2. SESSION STATE YÖNETİMİ & GEÇMİŞ TAKİBİ
# ==============================================================================

if "history_manager" not in st.session_state:
    st.session_state.history_manager = DataHistoryManager(max_history=15)

if "last_op_feedback" not in st.session_state:
    st.session_state.last_op_feedback = None


# ==============================================================================
# 3. VERİ YÜKLEME VE CACHING MOTORU
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_cached_sample_data() -> pd.DataFrame:
    """Örnek veri setini hafızada önbelleğe alır ve anında döner."""
    return DataProcessor.generate_sample_sales_data()


@st.cache_data(show_spinner=False)
def load_excel_cached(file_bytes: bytes, sheet_name: Optional[str], header_row: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Excel tablosunu hafızadaki bayt akışından hızlı ve hatasız yükler."""
    return DataProcessor.load_excel(file_bytes, sheet_name=sheet_name, header=header_row)


@st.cache_data(show_spinner=False)
def load_csv_cached(
    file_bytes: bytes,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    decimal: Optional[str] = None,
    header_row: int = 0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """CSV tablosunu akıllı kodlama, ayraç ve ondalık tespitiyle hatasız yükler."""
    return DataProcessor.load_csv(
        file_bytes=file_bytes,
        sep=sep,
        encoding=encoding,
        decimal=decimal,
        header=header_row
    )


# ==============================================================================
# 4. SIDEBAR & VERİ KAYNAĞI SEÇİMİ (ANINDA YÜKLENİR)
# ==============================================================================

with st.sidebar:
    st.markdown("### 📥 Veri Kaynağı")
    data_source = st.radio(
        "Veri Girişi Yöntemi",
        ["🎯 Hazır Örnek Veri Seti Kullan", "📁 Excel / CSV Dosyası Yükle (.xlsx, .xls, .csv)"],
        index=0
    )

    raw_df = None
    selected_sheet = None
    header_row = 0
    uploaded_file = None
    file_meta: Dict[str, Any] = {}

    if data_source == "📁 Excel / CSV Dosyası Yükle (.xlsx, .xls, .csv)":
        uploaded_file = st.file_uploader(
            "Excel veya CSV Dosyanızı Seçin",
            type=["xlsx", "xls", "xlsm", "csv", "tsv", "txt"],
            help="Microsoft Excel (.xlsx, .xls) veya CSV/TSV formatındaki tablolarınızı buraya sürükleyip bırakabilirsiniz."
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                fn_lower = uploaded_file.name.lower()
                is_csv = fn_lower.endswith((".csv", ".tsv", ".txt"))

                if is_csv:
                    selected_sheet = "CSV"

                    # Gelişmiş CSV Parametreleri (İsteğe bağlı manuel kontrol)
                    with st.expander("⚙️ Gelişmiş CSV / Metin Ayarları", expanded=False):
                        csv_sep_opt = st.selectbox(
                            "Ayraç (Delimiter)",
                            ["Otomatik Tespit", "Virgül (,)", "Noktalı Virgül (;)", "Tab (\\t)", "Dikey Çizgi (|)"],
                            index=0,
                            help="Dosyanızda sütunları ayıran karakter. 'Otomatik Tespit' önerilir."
                        )
                        csv_enc_opt = st.selectbox(
                            "Karakter Kodlaması (Encoding)",
                            ["Otomatik Tespit", "UTF-8", "UTF-8-BOM", "CP1254 (Türkçe Windows)", "ISO-8859-9", "Latin-1"],
                            index=0,
                            help="Türkçe karakterlerin doğru görünmesi için kodlama. 'Otomatik Tespit' akıllıca seçer."
                        )
                        csv_dec_opt = st.selectbox(
                            "Ondalık Ayracı (Decimal)",
                            ["Otomatik Tespit", "Nokta (.)", "Virgül (,)"],
                            index=0,
                            help="Sayısal ondalık ayracı."
                        )
                        header_row = st.number_input(
                            "Başlık (Header) Satır İndeksi",
                            min_value=0, max_value=20, value=0,
                            help="Sütun isimlerinin bulunduğu satır indeksi (genellikle 0)."
                        )

                    sep_map = {"Virgül (,)": ",", "Noktalı Virgül (;)": ";", "Tab (\\t)": "\t", "Dikey Çizgi (|)": "|"}
                    enc_map = {"UTF-8": "utf-8", "UTF-8-BOM": "utf-8-sig", "CP1254 (Türkçe Windows)": "cp1254", "ISO-8859-9": "iso-8859-9", "Latin-1": "latin-1"}
                    dec_map = {"Nokta (.)": ".", "Virgül (,)": ","}

                    custom_sep = sep_map.get(csv_sep_opt)
                    custom_enc = enc_map.get(csv_enc_opt)
                    custom_dec = dec_map.get(csv_dec_opt)

                    with st.spinner("CSV tablosu taranıyor, kodlama ve ayraçlar optimize ediliyor..."):
                        raw_df, file_meta = load_csv_cached(
                            file_bytes=file_bytes,
                            sep=custom_sep,
                            encoding=custom_enc,
                            decimal=custom_dec,
                            header_row=header_row
                        )
                else:
                    excel_obj = pd.ExcelFile(io.BytesIO(file_bytes))
                    sheet_names = excel_obj.sheet_names

                    if len(sheet_names) > 1:
                        selected_sheet = st.selectbox("📄 İncelenecek Çalışma Sayfası (Sheet)", sheet_names)
                    else:
                        selected_sheet = sheet_names[0]
                        st.info(f"📄 Çalışma Sayfası: **{selected_sheet}**")

                    header_row = st.number_input(
                        "Başlık (Header) Satır İndeksi",
                        min_value=0, max_value=20, value=0,
                        help="Sütun isimlerinin bulunduğu satır indeksi (genellikle 0)."
                    )

                    with st.spinner("Excel tablosu yükleniyor ve şema taranıyor..."):
                        raw_df, file_meta = load_excel_cached(file_bytes, selected_sheet, header_row)

                if raw_df is not None and not raw_df.empty:
                    NotificationHub.notify_success(f"'{uploaded_file.name}' yüklendi ({len(raw_df):,} satır × {len(raw_df.columns)} sütun).")
                else:
                    st.warning("⚠️ Seçilen dosyada okunabilir veri bulunamadı. Başlık satır indeksini veya ayraç ayarlarını kontrol edin.")

            except Exception as e:
                st.error(f"❌ Dosya okunurken bir hata oluştu:\n{e}")
                NotificationHub.notify_error(f"Dosya okuma hatası: {str(e)}")
                raw_df = None
    else:
        st.success("✅ 'Satış & Müşteri Performansı' örnek veri seti aktif.")
        raw_df = get_cached_sample_data()
        file_meta = {"encoding": "UTF-8", "rows": len(raw_df), "cols": len(raw_df.columns), "duration_ms": 0.0}

    st.markdown("---")
    st.markdown("### ⚙️ Görsel Ayarlar")
    theme_plot = st.selectbox("Grafik Renk Teması", ["plotly_dark", "plotly_white", "ggplot2", "seaborn", "simple_white"], index=0)


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

if "active_df" not in st.session_state or st.session_state.get("last_source") != current_source_sig:
    st.session_state.active_df = raw_df.copy()
    st.session_state.last_source = current_source_sig
    st.session_state.history_manager = DataHistoryManager(max_history=15)
    if "automl_result" in st.session_state:
        del st.session_state["automl_result"]

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
    st.markdown("### 📋 Veri Tablosu ve Detaylı Şema İnceleme")

    col_exp_1, col_exp_2 = st.columns([3, 1])
    with col_exp_1:
        search_term = st.text_input("🔍 Sütunlarda Filtrele / Ara:", placeholder="Aramak istediğiniz anahtar kelime...")
    with col_exp_2:
        row_limit = st.selectbox("Gösterilecek Satır Sayısı", [50, 100, 250, 500, "Tümü"], index=0)

    filtered_df_display = df.copy()
    if search_term:
        mask = filtered_df_display.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
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


# ==============================================================================
# SEKME 2: VERİ TEMİZLEME, MANİPÜLASYON & GERİ ALMA (UNDO/REDO)
# ==============================================================================
with tab_cleaning:
    st.markdown("### 🧹 Akıllı Veri Temizleme & Manipülasyon Stüdyosu")
    st.info("💡 Burada yaptığınız tüm işlemler anlık olarak aktif veri setine yansır ve işlem geçmişinde saklanır. İstediğiniz zaman geri alabilirsiniz.")

    # Geri Alma / İleri Alma Kontrol Çubuğu
    history_mgr = st.session_state.history_manager
    col_undo1, col_undo2, col_undo3 = st.columns([1, 1, 3])

    with col_undo1:
        if st.button("↩️ Son İşlemi Geri Al (Undo)", disabled=not history_mgr.can_undo(), width="stretch"):
            with st.spinner("Geri alınıyor..."):
                start_t = time.time()
                prev_df, rec = history_mgr.undo(df)
                st.session_state.active_df = prev_df
                duration = (time.time() - start_t) * 1000
                st.session_state.last_op_feedback = {
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
                st.session_state.active_df = next_df
                duration = (time.time() - start_t) * 1000
                st.session_state.last_op_feedback = {
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
                    # Validasyon
                    if clean_target_col in df.columns:
                        val_rep = InputValidator.validate_imputation(df[clean_target_col], impute_method, custom_fill_val)
                        if not val_rep.is_valid:
                            for issue in val_rep.get_errors():
                                st.error(f"❌ {issue.message}")
                            st.stop()

                    start_t = time.time()
                    with st.spinner("Eksik değerler dolduruluyor..."):
                        old_state = df.copy()
                        new_df, desc_msg = DataProcessor.apply_imputation(df, clean_target_col, impute_method, custom_fill_val)
                        history_mgr.record_change(old_state, new_df, "Eksik Değer Doldurma", desc_msg)
                        st.session_state.active_df = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state.last_op_feedback = {
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
                    new_df = DataProcessor.sanitize_dataframe(new_df)

                    desc_msg = f"{dropped} adet mükerrer satır silindi."
                    history_mgr.record_change(old_state, new_df, "Duplicate Temizliği", desc_msg)
                    st.session_state.active_df = new_df
                    duration = (time.time() - start_t) * 1000
                    st.session_state.last_op_feedback = {
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
                    new_df, dropped_rows, err = DataProcessor.apply_filter_rule(df, filter_col, filter_op, f_val1, f_val2)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        desc_msg = f"'{filter_col}' {filter_op} filtresi uygulandı ({dropped_rows} satır elendi)."
                        history_mgr.record_change(old_state, new_df, "Kural Filtreleme", desc_msg)
                        st.session_state.active_df = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state.last_op_feedback = {
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
                            new_df, out_cnt, desc_msg = DataProcessor.apply_outlier_action(
                                df, outlier_col, outlier_method, outlier_action
                            )
                            history_mgr.record_change(old_state, new_df, "Aykırı Değer Yönetimi", desc_msg)
                            st.session_state.active_df = new_df
                            duration = (time.time() - start_t) * 1000
                            st.session_state.last_op_feedback = {
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
                    new_df, err = DataProcessor.create_calculated_column(df, new_col_name, formula_input)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        desc_msg = f"'{new_col_name}' sütunu formülle başarıyla türetildi."
                        history_mgr.record_change(old_state, new_df, "Hesaplanan Sütun", desc_msg)
                        st.session_state.active_df = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state.last_op_feedback = {
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
                        new_df = DataProcessor.sanitize_dataframe(new_df)
                        desc_msg = f"{len(cols_to_drop)} sütun silindi."
                        history_mgr.record_change(old_state, new_df, "Sütun Silme", desc_msg)
                        st.session_state.active_df = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state.last_op_feedback = {
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
                    # Güvenlik ve veri kaybı analizi
                    val_rep, succ_rate = InputValidator.validate_type_cast(df[col_to_cast], target_type)
                    if not val_rep.is_valid:
                        for iss in val_rep.get_errors():
                            st.error(f"❌ {iss.message}")
                        st.stop()

                    start_t = time.time()
                    with st.spinner("Veri tipi dönüştürülüyor..."):
                        old_state = df.copy()
                        new_df, desc_msg = DataProcessor.apply_type_cast(df, col_to_cast, target_type)
                        history_mgr.record_change(old_state, new_df, "Tip Dönüştürme", desc_msg)
                        st.session_state.active_df = new_df
                        duration = (time.time() - start_t) * 1000
                        st.session_state.last_op_feedback = {
                            "action_name": "Tip Dönüştürme",
                            "description": desc_msg,
                            "duration_ms": duration
                        }
                        NotificationHub.notify_success(desc_msg)
                        st.rerun()

    st.markdown("---")
    if st.button("🔄 TÜM DEĞİŞİKLİKLERİ SIFIRLA VE İLK HALİNE DÖN", type="secondary", width="stretch"):
        with st.spinner("Veri seti ilk haline döndürülüyor..."):
            st.session_state.active_df = raw_df.copy()
            st.session_state.history_manager = DataHistoryManager(max_history=15)
            if "automl_result" in st.session_state:
                del st.session_state["automl_result"]
            st.session_state.last_op_feedback = {
                "action_name": "Sıfırlama",
                "description": "Veri seti orijinal haline geri döndürüldü.",
                "duration_ms": 10.0
            }
            NotificationHub.notify_info("Veri seti orijinal haline döndürüldü.")
            st.rerun()


# ==============================================================================
# SEKME 3: KEŞİFÇİ VERİ ANALİZİ (EDA)
# ==============================================================================
with tab_eda:
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


# ==============================================================================
# SEKME 4: GELİŞMİŞ İNTERAKTİF GÖRSELLEŞTİRME STÜDYOSU
# ==============================================================================
with tab_charts:
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
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

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

            # NaN ve negatif boyut koruması (Plotly marker size invalid element [nan] hatasını engeller)
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


# ==============================================================================
# SEKME 5: 1-TIK AUTOML & MAKİNE ÖĞRENMESİ STÜDYOSU
# ==============================================================================
with tab_stats_ml:
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
        rec_target_idx = AutoMLEngine.get_recommended_target_index(all_columns, df)

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
            # Validasyon Muhafızı
            val_report = MLGuard.validate_ml_setup(df, target_col, task_type, selected_features)
            if not val_report.is_valid:
                for err in val_report.get_errors():
                    st.error(f"❌ {err.message}")
                st.stop()

            for warn in val_report.get_warnings():
                st.warning(f"⚠️ {warn.message}")

            progress_bar = st.progress(0, text="Modeller eğitiliyor...")

            def progress_update(ratio, text):
                progress_bar.progress(ratio, text=text)

            try:
                start_t = time.time()
                result = AutoMLEngine.train_models(
                    df=df,
                    target_col=target_col,
                    task_type=task_type,
                    selected_features=selected_features,
                    test_size=test_size,
                    progress_cb=progress_update
                )
                progress_bar.empty()
                st.session_state["automl_result"] = result
                duration = (time.time() - start_t) * 1000
                NotificationHub.notify_success(f"AutoML tamamlandı! Şampiyon Model: {result['best_model_name']}")
            except Exception as ex:
                progress_bar.empty()
                st.error(f"❌ Eğitim sırasında hata oluştu: {str(ex)}")

        # Sonuçları Göster
        if "automl_result" in st.session_state:
            res = st.session_state["automl_result"]
            lead_df = res["leaderboard_df"].copy()
            best_m_name = res["best_model_name"]
            best_pipe = res["best_pipe"]
            is_classif = "Sınıflandırma" in res["task_type"]

            st.markdown("---")
            primary_metric_label = "F1-Score" if is_classif else "R² Skoru"
            primary_metric_val = f"%{lead_df.iloc[0]['F1-Score (%)']}" if is_classif else f"{lead_df.iloc[0]['R² Skoru']}"

            UIComponents.render_champion_card(
                best_model_name=best_m_name,
                target_col=res["target_col"],
                task_type=res["task_type"],
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
                try:
                    feat_names = []
                    if hasattr(res["preprocessor"], "get_feature_names_out"):
                        feat_names = list(res["preprocessor"].get_feature_names_out())
                    else:
                        feat_names = res["num_features"] + res["cat_features"]

                    model_obj = best_pipe.named_steps["model"]
                    importances = None

                    if hasattr(model_obj, "feature_importances_"):
                        importances = model_obj.feature_importances_
                    elif hasattr(model_obj, "coef_"):
                        importances = np.abs(model_obj.coef_)
                        if importances.ndim > 1:
                            importances = importances.mean(axis=0)

                    if importances is not None and len(importances) == len(feat_names):
                        imp_df = pd.DataFrame({
                            "Özellik": [f.replace("num__", "").replace("cat__", "") for f in feat_names],
                            "Önem Düzeyi": importances
                        }).sort_values(by="Önem Düzeyi", ascending=True).tail(15)

                        fig_imp = px.bar(
                            imp_df, x="Önem Düzeyi", y="Özellik", orientation="h",
                            title=f"{best_m_name} - En Önemli Değişkenler (Feature Importance)",
                            template=theme_plot, color="Önem Düzeyi", color_continuous_scale="Tealgrn"
                        )
                        st.plotly_chart(fig_imp, width="stretch")
                    else:
                        perm_res = permutation_importance(best_pipe, res["X_test"], res["y_test"], n_repeats=5, random_state=42)
                        imp_df = pd.DataFrame({
                            "Özellik": res["engineered_features"],
                            "Önem Düzeyi": np.maximum(0, perm_res.importances_mean)
                        }).sort_values(by="Önem Düzeyi", ascending=True).tail(15)

                        fig_imp = px.bar(
                            imp_df, x="Önem Düzeyi", y="Özellik", orientation="h",
                            title=f"{best_m_name} - Permütasyon Özellik Önemi",
                            template=theme_plot, color="Önem Düzeyi", color_continuous_scale="Purp"
                        )
                        st.plotly_chart(fig_imp, width="stretch")
                except Exception as ex_imp:
                    st.caption(f"Özellik önemi grafiği: {ex_imp}")

            with col_viz2:
                y_test_pred = best_pipe.predict(res["X_test"])
                if is_classif:
                    labels = np.unique(res["y_test"])
                    cm = confusion_matrix(res["y_test"], y_test_pred, labels=labels)
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
                    fig_reg = px.scatter(
                        x=res["y_test"], y=y_test_pred,
                        labels={"x": "Gerçek Değerler", "y": "Tahmin Edilen Değerler"},
                        title=f"{best_m_name} - Gerçek vs. Tahmin Regresyon Grafiği",
                        template=theme_plot
                    )
                    min_v, max_v = min(res["y_test"].min(), y_test_pred.min()), max(res["y_test"].max(), y_test_pred.max())
                    fig_reg.add_trace(go.Scatter(
                        x=[min_v, max_v], y=[min_v, max_v],
                        mode="lines", name="İdeal Tahmin (Y=X)",
                        line=dict(color="#EF4444", dash="dash")
                    ))
                    st.plotly_chart(fig_reg, width="stretch")

    # 5.2 CANLI TAHMİN SİMÜLATÖRÜ & TOPLU TAHMİN
    with ml_sub_2:
        st.markdown("#### 🔮 İnteraktif 'What-If' Tahmin Simülatörü & Toplu Tahmin")
        if "automl_result" not in st.session_state:
            st.info("👈 Tahmin yapabilmek için lütfen önce **'1. 1-Tık AutoML & Model Liderlik Tablosu'** sekmesinden modellerinizi eğitin.")
        else:
            res = st.session_state["automl_result"]
            best_pipe = res["best_pipe"]
            best_m_name = res["best_model_name"]
            selected_features = res["selected_features"]
            is_classif = "Sınıflandırma" in res["task_type"]

            sim_tab1, sim_tab2 = st.tabs(["⚡ Tekil Canlı Tahmin (What-If)", "📁 Toplu Tahmin (Batch Prediction)"])

            with sim_tab1:
                st.markdown(f"**Aktif Model:** `{best_m_name}` | **Hedef Değişken:** `{res['target_col']}`")
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
                        pred_val, prob_df, err = AutoMLEngine.predict_single(
                            best_pipe=best_pipe,
                            raw_inputs=raw_inputs,
                            engineered_features=res["engineered_features"],
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
                        target_batch_df, b_meta = DataProcessor.load_file(b_bytes, batch_file.name)
                        st.caption(f"Yüklenen dosya: **{len(target_batch_df):,}** satır × **{len(target_batch_df.columns)}** sütun.")
                    except Exception as b_err:
                        st.error(f"Dosya okunamadı: {b_err}")
                else:
                    if st.button("🧪 Mevcut Test Verisi ile Toplu Tahmin Çalıştır"):
                        target_batch_df = df.loc[res["X_test"].index, selected_features].copy()

                if target_batch_df is not None:
                    with st.spinner("Toplu tahminler hesaplanıyor..."):
                        out_batch_df, b_err = AutoMLEngine.predict_batch(
                            best_pipe=best_pipe,
                            target_batch_df=target_batch_df,
                            selected_features=selected_features,
                            engineered_features=res["engineered_features"]
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
                        st.session_state["anom_result"] = an_res
                        NotificationHub.notify_success(f"{an_res['total_anomalies']} anomali tespit edildi.")

            if "anom_result" in st.session_state:
                an_res = st.session_state["anom_result"]
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


# ==============================================================================
# SEKME 6: AKILLI İÇGÖRÜLER & AI
# ==============================================================================
with tab_insights:
    st.markdown("### 💡 Otomatik Akıllı İçgörüler & Yönetici Özeti")
    c_ins1, c_ins2 = st.columns([1, 1])

    with c_ins1:
        st.markdown("#### 🛡️ Veri Sağlığı ve Risk Değerlendirmesi")
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: {health_dict['color']}; margin-top:0;">{health_dict['grade']}</h3>
            <p><strong>Genel Sağlık Skoru:</strong> %{health_dict['score']}</p>
            <ul>
                <li><strong>Eksik Değer Oranı:</strong> %{health_dict['missing_pct']} ({health_dict['missing_cells']} hücre)</li>
                <li><strong>Duplicate Oranı:</strong> %{health_dict['dup_pct']} ({health_dict['dup_rows']} satır)</li>
                <li><strong>Toplam Boyut:</strong> {len(df)} satır x {len(df.columns)} sütun ({schema_report.metrics.get('memory_mb', 0)} MB)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_ins2:
        st.markdown("#### 💡 Öne Çıkan Algoritmik Bulgular")
        findings = []
        num_cols_ins = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols_ins) >= 2:
            corr_mat = df[num_cols_ins].corr(numeric_only=True).abs()
            corr_arr = corr_mat.to_numpy().copy()
            np.fill_diagonal(corr_arr, 0)
            corr_mat = pd.DataFrame(corr_arr, index=corr_mat.index, columns=corr_mat.columns)
            max_corr = corr_mat.max().max() if not corr_mat.empty else 0
            if max_corr > 0.7:
                c_pair = corr_mat.stack().idxmax()
                findings.append(f"🔗 **Güçlü İlişki:** `{c_pair[0]}` ve `{c_pair[1]}` değişkenleri arasında çok yüksek korelasyon (r = {max_corr:.2f}) tespit edildi.")

        if num_cols_ins:
            cv_series = (df[num_cols_ins].std() / df[num_cols_ins].mean()).abs().dropna()
            if not cv_series.empty:
                top_var_col = cv_series.idxmax()
                findings.append(f"📊 **En Yüksek Değişkenlik:** `{top_var_col}` sütunu veri setinde en yüksek dalgalanmaya (varyasyon katsayısına) sahip metriktir.")

        cat_cols_ins = df.select_dtypes(exclude=np.number).columns.tolist()
        if cat_cols_ins:
            top_cat = cat_cols_ins[0]
            top_val = df[top_cat].mode().iloc[0] if not df[top_cat].mode().empty else "Yok"
            val_pct = (df[top_cat] == top_val).mean() * 100
            findings.append(f"🏷️ **Baskın Kategori:** `{top_cat}` alanında en sık görülen değer **'{top_val}'** (tüm verinin %{val_pct:.1f}'i).")

        if not findings:
            findings.append("Veri setinde belirgin bir kritik aykırılık veya ekstrem ilişki saptanmadı.")

        for f in findings:
            st.info(f)

    st.markdown("---")
    c_gem1, c_gem2 = st.columns([2, 1])
    with c_gem1:
        ai_api_key = st.text_input("Gemini API Key (İsteğe Bağlı)", type="password", help="Kendi Google Gemini API anahtarınızı girerek derinlemesine analiz alabilirsiniz.")
    with c_gem2:
        gemini_model_choice = st.selectbox("🤖 Model Sürümü:", ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.1-pro"], index=0)

    if ai_api_key:
        user_question = st.text_area("Veri setiyle ilgili sormak istediğiniz soru veya rapor talebi:", placeholder="Örn: Bu verideki en karlı müşteri segmenti hangisi ve nelere dikkat edilmeli?")
        if st.button("🚀 Gemini AI ile Analiz Et", width="stretch"):
            with st.spinner(f"{gemini_model_choice} verinizi analiz ediyor..."):
                try:
                    from google import genai
                    client = genai.Client(api_key=ai_api_key)
                    summary_text = f"""
                    Veri Seti Bilgileri:
                    Satır: {len(df)}, Sütun: {len(df.columns)}
                    Sayısal Özet:
                    {df.describe().to_string()}
                    
                    Kullanıcı Sorusu:
                    {user_question if user_question else 'Bu veri seti için üst yöneticiye sunulacak 5 maddelik stratejik içgörü raporu oluştur.'}
                    """
                    response = client.models.generate_content(model=gemini_model_choice, contents=summary_text)
                    st.markdown("##### 📄 AI Yönetici Özeti & Cevap")
                    st.markdown(response.text)
                    NotificationHub.notify_success("AI analizi başarıyla tamamlandı.")
                except Exception as ex:
                    st.error(f"Gemini API çağrısı sırasında hata oluştu: {ex}")


# ==============================================================================
# SEKME 7: DIŞA AKTARMA & RAPORLAMA
# ==============================================================================
with tab_export:
    st.markdown("### 💾 Temizlenmiş Veriyi ve Raporları Dışa Aktarın")

    col_exp_1, col_exp_2 = st.columns(2)

    with col_exp_1:
        st.markdown("#### 📥 Temizlenmiş Veri İndirme")
        st.write("Tüm manipülasyon adımlarından geçmiş güncel tabloyu aşağıdaki formatlarda indirebilirsiniz:")

        # Excel İndir
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Temizlenmis_Veri")
        excel_data = excel_buffer.getvalue()

        st.download_button(
            label="📊 Güncel Veriyi Excel (.xlsx) Olarak İndir",
            data=excel_data,
            file_name=f"temizlenmis_veri_{int(time.time())}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

        # CSV İndir
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📄 Güncel Veriyi CSV (.csv) Olarak İndir",
            data=csv_data,
            file_name=f"temizlenmis_veri_{int(time.time())}.csv",
            mime="text/csv",
            width="stretch"
        )

    with col_exp_2:
        st.markdown("#### 📑 Otomatik Analiz Raporu İndirme")
        st.write("Platform tarafından derlenen EDA, validasyon ve istatistiksel özet raporunu Markdown olarak alın:")

        desc_table = "Sayısal sütun bulunmuyor."
        if not num_df.empty:
            try:
                desc_table = num_df.describe().to_markdown()
            except Exception:
                desc_table = f"```\n{num_df.describe().to_string()}\n```"

        ml_summary = "Henüz AutoML modeli çalıştırılmadı."
        if "automl_result" in st.session_state:
            res = st.session_state["automl_result"]
            ml_summary = f"Şampiyon Model: {res['best_model_name']} (Hedef: {res['target_col']})"

        report_md = f"""# DataVortex Veri Bilimi, Validasyon ve Analiz Raporu
Tarih: {time.strftime('%d.%m.%Y %H:%M')}

## 1. Genel Veri ve Sağlık Özeti
- **Toplam Satır Sayısı:** {len(df)}
- **Toplam Sütun Sayısı:** {len(df.columns)}
- **Eksik Hücre Sayısı:** {schema_report.metrics.get('missing_cells', 0)} (%{schema_report.metrics.get('missing_pct', 0.0)})
- **Duplicate Kayıt Sayısı:** {schema_report.metrics.get('dup_rows', 0)}
- **Veri Kalite Skoru:** %{schema_report.score} ({schema_report.grade})
- **Makine Öğrenmesi Durumu:** {ml_summary}

## 2. Sayısal Değişken İstatistikleri
{desc_table}

---
*Rapor DataVortex Akıllı Excel Veri Bilimi & AutoML Platformu tarafından otomatik olarak üretilmiştir.*
"""
        st.download_button(
            label="📑 Analiz Raporunu Markdown (.md) Olarak İndir",
            data=report_md.encode("utf-8"),
            file_name=f"veri_analiz_raporu_{int(time.time())}.md",
            mime="text/markdown",
            width="stretch"
        )


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