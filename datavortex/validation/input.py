"""
DataVortex — Veri Manipülasyonu ve Kullanıcı Girdi Denetleyicisi
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .models import ValidationIssue, ValidationReport, ValidationSeverity


class InputValidator:
    """Kullanıcının veri manipülasyonu yaparken girdiği değerlerin tip ve mantık kurallarını denetler."""

    @staticmethod
    def validate_imputation(series: pd.Series, method: str, custom_val: Optional[str] = None) -> ValidationReport:
        """Eksik değer doldurma parametrelerinin tutarlılığını denetler."""
        report = ValidationReport(is_valid=True)

        if series is None or series.empty:
            report.add_issue(ValidationIssue(
                field="Sütun",
                message="Seçilen sütunda veri bulunmuyor.",
                severity=ValidationSeverity.ERROR,
                code="EMPTY_SERIES"
            ))
            return report

        is_num = pd.api.types.is_numeric_dtype(series)

        if method in ["Ortalama ile Doldur (Mean)", "Medyan ile Doldur (Median)"] and not is_num:
            report.add_issue(ValidationIssue(
                field=series.name or "Sütun",
                message=f"'{method}' yöntemi yalnızca sayısal sütunlarda kullanılabilir. Seçilen sütun metin/kategoriktir.",
                severity=ValidationSeverity.ERROR,
                code="INVALID_NUMERIC_METHOD_ON_TEXT",
                suggestion="En Çok Tekrar Eden (Mode) veya Sabit Değer ile Doldur yöntemini seçin."
            ))

        if method == "Sabit Değer ile Doldur":
            if custom_val is None or str(custom_val).strip() == "":
                report.add_issue(ValidationIssue(
                    field="Sabit Değer",
                    message="Doldurulacak sabit değer boş bırakılamaz.",
                    severity=ValidationSeverity.ERROR,
                    code="EMPTY_CONSTANT_VALUE"
                ))
            elif is_num:
                try:
                    float(str(custom_val).strip().replace(",", "."))
                except ValueError:
                    report.add_issue(ValidationIssue(
                        field="Sabit Değer",
                        message=f"'{custom_val}' değeri sayısal bir sütun için geçerli bir sayı değildir.",
                        severity=ValidationSeverity.ERROR,
                        code="TYPE_MISMATCH_NUMERIC_CONSTANT",
                        suggestion="Örn: 0, 100 veya 3.14 gibi geçerli bir sayı girin."
                    ))

        return report

    @staticmethod
    def validate_outlier_bounds(lower: float, upper: float, series: pd.Series) -> ValidationReport:
        """Aykırı değer alt/üst sınır mantığını doğrular."""
        report = ValidationReport(is_valid=True)
        if lower >= upper:
            report.add_issue(ValidationIssue(
                field="Aykırı Sınırları",
                message=f"Alt sınır ({lower:.2f}) üst sınırdan ({upper:.2f}) büyük veya eşit olamaz.",
                severity=ValidationSeverity.ERROR,
                code="INVALID_BOUND_RANGE"
            ))
        if series.std() == 0 or np.isnan(series.std()):
            report.add_issue(ValidationIssue(
                field=series.name or "Sütun",
                message="Sütun varyansı sıfır olduğu için aykırı değer hesaplanamaz.",
                severity=ValidationSeverity.WARNING,
                code="ZERO_VARIANCE_FOR_OUTLIER"
            ))
        return report

    @staticmethod
    def validate_type_cast(series: pd.Series, target_type: str) -> Tuple[ValidationReport, float]:
        """
        Bir sütunun hedef tipe dönüştürülme güvenliğini analiz eder.
        Dönüşümde kaybolacak / NaN olacak veri oranını hesaplar.
        """
        report = ValidationReport(is_valid=True)
        valid_non_null = series.dropna()
        if valid_non_null.empty:
            return report, 100.0

        total_count = len(valid_non_null)
        success_count = 0

        try:
            if "Sayısal" in target_type or "Float" in target_type:
                converted = pd.to_numeric(valid_non_null, errors="coerce")
                success_count = int(converted.notna().sum())
            elif "Tamsayı" in target_type or "Integer" in target_type:
                converted = pd.to_numeric(valid_non_null, errors="coerce")
                success_count = int(converted.notna().sum())
            elif "Tarih" in target_type or "Datetime" in target_type:
                converted = pd.to_datetime(valid_non_null, errors="coerce")
                success_count = int(converted.notna().sum())
            else:
                success_count = total_count
        except Exception:
            success_count = 0

        success_rate = (success_count / total_count) * 100.0

        if success_rate < 50.0:
            report.add_issue(ValidationIssue(
                field=series.name or "Sütun",
                message=f"Dönüştürme işlemi verilerin %{100 - success_rate:.1f}'ini geçersiz kılacak (NaN yapacak).",
                severity=ValidationSeverity.ERROR,
                code="HIGH_DATA_LOSS_ON_CAST",
                suggestion="Lütfen sütunun orijinal içeriğini kontrol edin."
            ))
        elif success_rate < 95.0:
            report.add_issue(ValidationIssue(
                field=series.name or "Sütun",
                message=f"Dönüştürme sırasında bazı satırlar (%{100 - success_rate:.1f}) dönüştürülemeyip NaN olacaktır.",
                severity=ValidationSeverity.WARNING,
                code="PARTIAL_DATA_LOSS_ON_CAST"
            ))

        return report, success_rate
