"""
DataVortex — Sol Menü (Sidebar) ve Veri Kaynağı Seçim Arayüzü
"""

import io
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from datavortex.core.config import (
    DEFAULT_THEMES,
    DELIMITER_MAP,
    ENCODING_MAP,
    DECIMAL_MAP,
    SUPPORTED_FILE_EXTENSIONS,
)
from datavortex.core.cache import (
    get_cached_sample_data,
    load_csv_cached,
    load_excel_cached,
)
from datavortex.data.io import generate_sample_sales_data
from datavortex.ui.components import NotificationHub


def render_sidebar() -> Tuple[
    Optional[pd.DataFrame],
    Dict[str, Any],
    Any,
    Optional[str],
    int,
    str,
    str
]:
    """
    Sol menüyü render eder ve seçilen veri kümesiyle ilgili bilgileri döner.
    Döner: (raw_df, file_meta, uploaded_file, selected_sheet, header_row, theme_plot, data_source)
    """
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
                type=SUPPORTED_FILE_EXTENSIONS,
                help="Microsoft Excel (.xlsx, .xls) veya CSV/TSV formatındaki tablolarınızı buraya sürükleyip bırakabilirsiniz."
            )

            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.getvalue()
                    fn_lower = uploaded_file.name.lower()
                    is_csv = fn_lower.endswith((".csv", ".tsv", ".txt"))

                    if is_csv:
                        selected_sheet = "CSV"

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

                        custom_sep = DELIMITER_MAP.get(csv_sep_opt)
                        custom_enc = ENCODING_MAP.get(csv_enc_opt)
                        custom_dec = DECIMAL_MAP.get(csv_dec_opt)

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
            raw_df = get_cached_sample_data(generate_sample_sales_data)
            file_meta = {"encoding": "UTF-8", "rows": len(raw_df), "cols": len(raw_df.columns), "duration_ms": 0.0}

        st.markdown("---")
        st.markdown("### ⚙️ Görsel Ayarlar")
        theme_plot = st.selectbox("Grafik Renk Teması", DEFAULT_THEMES, index=0)

    return raw_df, file_meta, uploaded_file, selected_sheet, header_row, theme_plot, data_source
