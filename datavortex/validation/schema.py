"""
DataVortex — Şema, Dosya Bütünlüğü ve DataFrame Kalite Denetleyicisi
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .models import ValidationIssue, ValidationReport, ValidationSeverity


class SchemaValidator:
    """Veri çerçevelerini, sütun başlıklarını ve serileştirme uyumluluğunu doğrular."""

    DANGEROUS_PATTERNS = [
        r"<.*?>", r"javascript:", r"onerror=", r"onload=", r"eval\(",
        r"drop\s+table", r"insert\s+into", r"delete\s+from", r"--", r";--",
        r"system\(", r"__import__"
    ]

    @classmethod
    def sanitize_column_names(cls, columns: List[Any]) -> Tuple[List[str], List[ValidationIssue]]:
        """
        Sütun isimlerindeki boşlukları, özel karakterleri, tekrarları ve potansiyel injection kalıplarını temizler.
        """
        clean_cols = []
        seen: Dict[str, int] = {}
        issues: List[ValidationIssue] = []

        for idx, col in enumerate(columns):
            raw_str = str(col).strip().replace("\n", " ").replace("\r", "")
            original_str = raw_str

            # Boş veya adsız sütun kontrolü
            if not raw_str or raw_str.lower().startswith("unnamed:") or raw_str.lower() == "nan":
                raw_str = f"Sütun_{idx + 1}"
                issues.append(ValidationIssue(
                    field=f"Sütun {idx + 1}",
                    message=f"'{original_str}' adlı tanımsız sütun '{raw_str}' olarak adlandırıldı.",
                    severity=ValidationSeverity.INFO,
                    code="UNNAMED_COLUMN_RENAMED",
                    suggestion="Excel dosyanızda sütun başlıklarını doldurmanız önerilir."
                ))

            # Güvenlik / Injection kontrolü
            for pattern in cls.DANGEROUS_PATTERNS:
                if re.search(pattern, raw_str, re.IGNORECASE):
                    safe_replacement = re.sub(r"[^\w\s\-.]", "_", raw_str)
                    issues.append(ValidationIssue(
                        field=raw_str,
                        message=f"'{raw_str}' sütun isminde potansiyel güvenlik riski oluşturan karakterler temizlendi.",
                        severity=ValidationSeverity.WARNING,
                        code="INJECTION_RISK_SANITIZED",
                        suggestion=f"Yeni isim: '{safe_replacement}'"
                    ))
                    raw_str = safe_replacement
                    break

            # Mükerrer sütun adı kontrolü
            if raw_str in seen:
                seen[raw_str] += 1
                unique_name = f"{raw_str}_{seen[raw_str]}"
                issues.append(ValidationIssue(
                    field=raw_str,
                    message=f"Mükerrer sütun ismi tespit edildi: '{raw_str}' -> '{unique_name}' olarak yeniden adlandırıldı.",
                    severity=ValidationSeverity.WARNING,
                    code="DUPLICATE_COLUMN_NAME",
                    suggestion="Tablonuzda aynı isimde birden fazla sütun bulunmamalıdır."
                ))
                clean_cols.append(unique_name)
            else:
                seen[raw_str] = 0
                clean_cols.append(raw_str)

        return clean_cols, issues

    @classmethod
    def validate_csv_integrity(cls, file_bytes: bytes, filename: str = "data.csv") -> ValidationReport:
        """
        Ham CSV bayt akışının dosya bütünlüğünü, satır ayraç tutarlılığını ve kodlama sağlığını denetler.
        """
        report = ValidationReport(is_valid=True)
        if not file_bytes:
            report.add_issue(ValidationIssue(
                field=filename,
                message="CSV dosyası tamamen boş (0 bayt).",
                severity=ValidationSeverity.CRITICAL,
                code="EMPTY_CSV_FILE"
            ))
            report.score = 0.0
            report.grade = "F (Boş Dosya)"
            return report

        # 1. Kodlama & Bozuk Karakter (Mojibake / Replacement Char) Denetimi
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("cp1254")
                report.add_issue(ValidationIssue(
                    field="Karakter Kodlaması",
                    message="Dosya UTF-8 standardında değil, CP1254 (Türkçe Windows) olarak algılandı.",
                    severity=ValidationSeverity.INFO,
                    code="NON_UTF8_ENCODING",
                    suggestion="UTF-8 veya UTF-8-BOM formatında kaydetmeniz önerilir."
                ))
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="replace")
                report.add_issue(ValidationIssue(
                    field="Karakter Kodlaması",
                    message="Dosya kodlamasında bilinmeyen baytlar tespit edildi (Latin-1 ile kurtarıldı).",
                    severity=ValidationSeverity.WARNING,
                    code="CORRUPTED_ENCODING"
                ))

        if "\ufffd" in text:
            bad_char_count = text.count("\ufffd")
            report.add_issue(ValidationIssue(
                field="Karakter Bütünlüğü",
                message=f"Metin içerisinde {bad_char_count} adet bozuk/tanımsız karakter (\ufffd) tespit edildi.",
                severity=ValidationSeverity.WARNING,
                code="MOJIBAKE_DETECTED",
                affected_count=bad_char_count
            ))

        # 2. Satır bazlı ayraç tutarlılığı (Jagged / Asimetrik satır denetimi)
        lines = [ln for ln in text.splitlines() if ln.strip()][:100]
        if len(lines) > 1:
            delims = [",", ";", "\t", "|"]
            best_delim = ","
            best_cnt = 0
            for d in delims:
                c = lines[0].count(d)
                if c > best_cnt:
                    best_cnt = c
                    best_delim = d

            if best_cnt > 0:
                header_cols = lines[0].count(best_delim)
                jagged_lines = 0
                for idx, ln in enumerate(lines[1:], start=2):
                    ln_cols = ln.count(best_delim)
                    if abs(ln_cols - header_cols) > 2:
                        jagged_lines += 1

                if jagged_lines > 0:
                    report.add_issue(ValidationIssue(
                        field="Satır Tutarlılığı (Jagged Rows)",
                        message=f"İlk 100 satırın {jagged_lines} tanesinde başlık sütun sayısından farklı ayraç sayısı tespit edildi.",
                        severity=ValidationSeverity.WARNING,
                        code="JAGGED_CSV_ROWS",
                        suggestion="Satırlarda fazladan tırnak veya kaçış karakteri (escape) gerekebilir."
                    ))

        # Kalite skoru
        report.score = 100.0 - (len(report.get_warnings()) * 15.0) - (len(report.get_errors()) * 40.0)
        report.score = max(0.0, min(100.0, report.score))
        if report.score >= 90:
            report.grade = "A+ (Mükemmel)"
        elif report.score >= 75:
            report.grade = "A (İyi)"
        else:
            report.grade = "B/C (Uyarılar Var)"
        return report

    @classmethod
    def validate_dataframe(cls, df: Optional[pd.DataFrame]) -> ValidationReport:
        """Bir pandas DataFrame'i baştan uca denetler."""
        report = ValidationReport(is_valid=True)

        if df is None:
            report.add_issue(ValidationIssue(
                field="Genel",
                message="Veri seti yüklenemedi (None nesnesi).",
                severity=ValidationSeverity.CRITICAL,
                code="DATAFRAME_NONE"
            ))
            report.score = 0.0
            report.grade = "F (Geçersiz)"
            return report

        if df.empty:
            report.add_issue(ValidationIssue(
                field="Genel",
                message="Veri seti tamamen boş (0 satır veya 0 sütun).",
                severity=ValidationSeverity.CRITICAL,
                code="DATAFRAME_EMPTY"
            ))
            report.score = 0.0
            report.grade = "F (Boş Veri)"
            return report

        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = max(df.size, 1)

        # 1. Eksik hücre kontrolü
        missing_cells = int(df.isna().sum().sum())
        missing_pct = (missing_cells / total_cells) * 100.0

        if missing_pct > 50.0:
            report.add_issue(ValidationIssue(
                field="Eksik Veri",
                message=f"Veri setinin %{missing_pct:.1f}'i boş hücrelerden oluşuyor.",
                severity=ValidationSeverity.WARNING,
                code="HIGH_MISSING_RATIO",
                suggestion="Veri Temizleme sekmesinden eksik değer doldurma işlemlerini uygulayın.",
                affected_count=missing_cells
            ))
        elif missing_cells > 0:
            report.add_issue(ValidationIssue(
                field="Eksik Veri",
                message=f"Toplam {missing_cells} hücrede (%{missing_pct:.2f}) eksik değer var.",
                severity=ValidationSeverity.INFO,
                code="MISSING_CELLS_FOUND",
                affected_count=missing_cells
            ))

        # 2. Mükerrer satır kontrolü
        dup_rows = int(df.duplicated().sum())
        dup_pct = (dup_rows / max(total_rows, 1)) * 100.0
        if dup_rows > 0:
            sev = ValidationSeverity.WARNING if dup_pct > 15.0 else ValidationSeverity.INFO
            report.add_issue(ValidationIssue(
                field="Mükerrer Satır",
                message=f"{dup_rows} adet (%{dup_pct:.1f}) mükerrer (duplicate) satır bulundu.",
                severity=sev,
                code="DUPLICATE_ROWS_FOUND",
                suggestion="Mükerrer Kayıt Temizliği modülünü kullanarak tekilleştirin.",
                affected_count=dup_rows
            ))

        # 3. Sonsuz değer (Infinite) kontrolü
        numeric_cols = df.select_dtypes(include=np.number).columns
        inf_count = 0
        for col in numeric_cols:
            inf_in_col = int(np.isinf(df[col]).sum())
            if inf_in_col > 0:
                inf_count += inf_in_col
                report.add_issue(ValidationIssue(
                    field=col,
                    message=f"'{col}' sütununda {inf_in_col} adet sonsuz (Inf / -Inf) değer var.",
                    severity=ValidationSeverity.ERROR,
                    code="INFINITE_VALUES_DETECTED",
                    suggestion="Sonsuz değerler model eğitiminde hataya yol açar. Temizleyin veya kırpın.",
                    affected_count=inf_in_col
                ))

        # 4. Tek değerli (Sıfır Varyanslı) sütun kontrolü
        for col in df.columns:
            non_null_s = df[col].dropna()
            if non_null_s.nunique() <= 1 and len(non_null_s) > 0:
                report.add_issue(ValidationIssue(
                    field=col,
                    message=f"'{col}' sütunu tek bir sabit değer içeriyor (Sıfır Varyans).",
                    severity=ValidationSeverity.WARNING,
                    code="ZERO_VARIANCE_COLUMN",
                    suggestion="Analize katkı sağlamayacağı için bu sütunu silebilirsiniz."
                ))

        # Kalite skoru hesaplama
        score = 100.0 - (missing_pct * 0.5) - (dup_pct * 0.3) - (min(inf_count, 20) * 1.0)
        score = max(0.0, min(100.0, round(score, 1)))

        if score >= 90:
            grade = "A+ (Mükemmel)"
        elif score >= 80:
            grade = "A (Çok İyi)"
        elif score >= 70:
            grade = "B (İyi / Kabul Edilebilir)"
        elif score >= 50:
            grade = "C (Temizlik Gerekli)"
        else:
            grade = "D/F (Kritik Sorunlar Var)"

        report.score = score
        report.grade = grade
        report.metrics = {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "missing_cells": missing_cells,
            "missing_pct": round(missing_pct, 2),
            "dup_rows": dup_rows,
            "dup_pct": round(dup_pct, 2),
            "inf_count": inf_count,
            "num_cols_cnt": len(numeric_cols),
            "cat_cols_cnt": total_cols - len(numeric_cols),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        return report
