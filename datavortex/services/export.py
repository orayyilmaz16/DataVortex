"""
DataVortex — Dışa Aktarma, Serileştirme ve Rapor Üretim Servisi (Export)
"""

import io
import time
from typing import Any, Dict, Optional
import pandas as pd


def export_dataframe_to_excel(df: pd.DataFrame, sheet_name: str = "Temizlenmis_Veri") -> bytes:
    """DataFrame'i Excel (.xlsx) bayt dizisine dönüştürür."""
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return excel_buffer.getvalue()


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """DataFrame'i UTF-8-BOM kodlu CSV bayt dizisine dönüştürür."""
    return df.to_csv(index=False).encode("utf-8-sig")


def generate_markdown_report(
    df: pd.DataFrame,
    schema_report: Any,
    automl_result: Optional[Dict[str, Any]] = None
) -> str:
    """EDA, validasyon ve ML özetini Markdown raporu olarak derler."""
    num_df = df.select_dtypes(include="number")
    desc_table = "Sayısal sütun bulunmuyor."
    if not num_df.empty:
        try:
            desc_table = num_df.describe().to_markdown()
        except Exception:
            desc_table = f"```\n{num_df.describe().to_string()}\n```"

    ml_summary = "Henüz AutoML modeli çalıştırılmadı."
    if automl_result is not None:
        best_name = automl_result.get("best_model_name", "Bilinmiyor")
        target_c = automl_result.get("target_col", "Bilinmiyor")
        ml_summary = f"Şampiyon Model: {best_name} (Hedef: {target_c})"

    missing_cnt = schema_report.metrics.get("missing_cells", 0) if hasattr(schema_report, "metrics") else 0
    missing_pct = schema_report.metrics.get("missing_pct", 0.0) if hasattr(schema_report, "metrics") else 0.0
    dup_rows = schema_report.metrics.get("dup_rows", 0) if hasattr(schema_report, "metrics") else 0
    score = getattr(schema_report, "score", 100.0)
    grade = getattr(schema_report, "grade", "A+ (Mükemmel)")

    report_md = f"""# DataVortex Veri Bilimi, Validasyon ve Analiz Raporu
Tarih: {time.strftime('%d.%m.%Y %H:%M')}

## 1. Genel Veri ve Sağlık Özeti
- **Toplam Satır Sayısı:** {len(df)}
- **Toplam Sütun Sayısı:** {len(df.columns)}
- **Eksik Hücre Sayısı:** {missing_cnt} (%{missing_pct})
- **Duplicate Kayıt Sayısı:** {dup_rows}
- **Veri Kalite Skoru:** %{score} ({grade})
- **Makine Öğrenmesi Durumu:** {ml_summary}

## 2. Sayısal Değişken İstatistikleri
{desc_table}

---
*Rapor DataVortex Akıllı Excel Veri Bilimi & AutoML Platformu tarafından otomatik olarak üretilmiştir.*
"""
    return report_md
