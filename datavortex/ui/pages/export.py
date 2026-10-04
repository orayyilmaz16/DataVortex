"""
DataVortex — Sayfa 7: Dışa Aktarma ve Analiz Raporlama (Export)
"""

import time
from typing import Any
import pandas as pd
import streamlit as st

from datavortex.core.state import get_automl_result
from datavortex.services.export import (
    export_dataframe_to_csv,
    export_dataframe_to_excel,
    generate_markdown_report,
)


def render_export_page(df: pd.DataFrame, schema_report: Any) -> None:
    """Temizlenmiş verileri (Excel, CSV) ve otomatik Markdown raporunu indirme arayüzünü render eder."""
    st.markdown("### 💾 Temizlenmiş Veriyi ve Raporları Dışa Aktarın")

    col_exp_1, col_exp_2 = st.columns(2)

    with col_exp_1:
        st.markdown("#### 📥 Temizlenmiş Veri İndirme")
        st.write("Tüm manipülasyon adımlarından geçmiş güncel tabloyu aşağıdaki formatlarda indirebilirsiniz:")

        # Excel İndir
        excel_data = export_dataframe_to_excel(df, sheet_name="Temizlenmis_Veri")
        st.download_button(
            label="📊 Güncel Veriyi Excel (.xlsx) Olarak İndir",
            data=excel_data,
            file_name=f"temizlenmis_veri_{int(time.time())}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

        # CSV İndir
        csv_data = export_dataframe_to_csv(df)
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

        automl_res = get_automl_result()
        report_md = generate_markdown_report(df, schema_report, automl_res)

        st.download_button(
            label="📑 Analiz Raporunu Markdown (.md) Olarak İndir",
            data=report_md.encode("utf-8"),
            file_name=f"veri_analiz_raporu_{int(time.time())}.md",
            mime="text/markdown",
            width="stretch"
        )
