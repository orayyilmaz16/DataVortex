"""
DataVortex — Makine Öğrenmesi Öncesi Hazırlık ve Güvenlik Muhafızı
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from .models import ValidationIssue, ValidationReport, ValidationSeverity


class MLGuard:
    """AutoML eğitiminden ve tahminlerden önce veri gereksinimlerini sıfır hata için doğrular."""

    @staticmethod
    def validate_ml_setup(
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        selected_features: List[str]
    ) -> ValidationReport:
        """Hedef değişken ve özelliklerin makine öğrenmesine hazır olup olmadığını tam denetler."""
        report = ValidationReport(is_valid=True)

        if df is None or df.empty:
            report.add_issue(ValidationIssue(
                field="Veri",
                message="Veri seti boş.",
                severity=ValidationSeverity.CRITICAL,
                code="EMPTY_DATASET"
            ))
            return report

        # 1. Hedef Değişken Kontrolleri
        if not target_col or target_col not in df.columns:
            report.add_issue(ValidationIssue(
                field="Hedef Değişken",
                message=f"Hedef değişken '{target_col}' tabloda bulunamadı.",
                severity=ValidationSeverity.CRITICAL,
                code="TARGET_NOT_FOUND"
            ))
            return report

        target_series = df[target_col].dropna()
        valid_rows_count = len(target_series)

        if valid_rows_count < 15:
            report.add_issue(ValidationIssue(
                field=target_col,
                message=f"Hedef değişkende yalnızca {valid_rows_count} geçerli satır var. Model eğitimi için en az 15 satır gereklidir.",
                severity=ValidationSeverity.CRITICAL,
                code="INSUFFICIENT_SAMPLES"
            ))

        # Tarih kontrolü
        is_date = pd.api.types.is_datetime64_any_dtype(target_series) or any(
            t in target_col.lower() for t in ["tarih", "date", "timestamp", "time"]
        )
        if is_date:
            report.add_issue(ValidationIssue(
                field=target_col,
                message=f"'{target_col}' bir Tarih/Zaman sütunudur. Zaman alanları model hedefi olamaz.",
                severity=ValidationSeverity.CRITICAL,
                code="DATE_TARGET_NOT_ALLOWED",
                suggestion="Kategorik (örn: Memnuniyet) veya Sürekli Sayısal (örn: Gelir) bir hedef seçin."
            ))

        # Göreve özgü kontroller
        n_unique_target = target_series.nunique()
        is_num = pd.api.types.is_numeric_dtype(target_series)

        if "Sınıflandırma" in task_type or "Classification" in task_type:
            if n_unique_target < 2:
                report.add_issue(ValidationIssue(
                    field=target_col,
                    message=f"Sınıflandırma için hedef değişkende en az 2 farklı sınıf olmalıdır. Mevcut sınıf sayısı: {n_unique_target}.",
                    severity=ValidationSeverity.CRITICAL,
                    code="SINGLE_CLASS_TARGET"
                ))
            elif n_unique_target > 30:
                report.add_issue(ValidationIssue(
                    field=target_col,
                    message=f"Hedef değişkende {n_unique_target} farklı değer var. 30'dan fazla sınıf için Regresyon görevi önerilir.",
                    severity=ValidationSeverity.ERROR,
                    code="TOO_MANY_CLASSES_FOR_CLASSIFICATION",
                    suggestion="Görevi 'Regresyon' olarak değiştirin veya hedef değişkeni gruplayın."
                ))
        else:
            # Regresyon
            if not is_num:
                report.add_issue(ValidationIssue(
                    field=target_col,
                    message=f"Regresyon modelleri yalnızca sayısal (numeric) hedefleri tahmin edebilir. '{target_col}' sütunu metin/kategoriktir.",
                    severity=ValidationSeverity.CRITICAL,
                    code="NON_NUMERIC_REGRESSION_TARGET",
                    suggestion="Görevi 'Sınıflandırma' olarak değiştirin veya sayısal bir hedef sütun seçin."
                ))
            elif n_unique_target <= 2:
                report.add_issue(ValidationIssue(
                    field=target_col,
                    message=f"'{target_col}' yalnızca {n_unique_target} farklı sayısal değer içeriyor (örn: 0 ve 1). Bu değişken için Sınıflandırma daha uygundur.",
                    severity=ValidationSeverity.WARNING,
                    code="BINARY_TARGET_FOR_REGRESSION"
                ))

        # 2. Özellik (Features) Kontrolleri
        features = [f for f in selected_features if f != target_col]
        if not features:
            report.add_issue(ValidationIssue(
                field="Bağımsız Değişkenler",
                message="Modele girecek en az 1 adet bağımsız değişken (özellik) seçilmelidir.",
                severity=ValidationSeverity.CRITICAL,
                code="NO_FEATURES_SELECTED"
            ))

        # Özellikler arası aşırı korelasyon (Multicollinearity)
        num_feats = df[features].select_dtypes(include=np.number).columns.tolist()
        if len(num_feats) >= 2:
            corr_mat = df[num_feats].corr().abs()
            corr_arr = corr_mat.to_numpy().copy()
            np.fill_diagonal(corr_arr, 0)
            corr = pd.DataFrame(corr_arr, index=corr_mat.index, columns=corr_mat.columns)
            high_corr_pairs = []
            for i in range(len(num_feats)):
                for j in range(i + 1, len(num_feats)):
                    c_val = corr.iloc[i, j]
                    if c_val > 0.98:
                        high_corr_pairs.append((num_feats[i], num_feats[j], round(c_val, 3)))

            for f1, f2, r in high_corr_pairs:
                report.add_issue(ValidationIssue(
                    field=f"{f1} ⚡ {f2}",
                    message=f"'{f1}' ve '{f2}' arasında aşırı yüksek korelasyon (r = {r}) var. Biri elenebilir.",
                    severity=ValidationSeverity.INFO,
                    code="HIGH_MULTICOLLINEARITY"
                ))

        return report

    @staticmethod
    def validate_what_if_inputs(raw_inputs: Dict[str, Any], schema_types: Dict[str, str]) -> ValidationReport:
        """Canlı simülatördeki kullanıcı girdilerinin geçerliliğini doğrular."""
        report = ValidationReport(is_valid=True)
        for feat, val in raw_inputs.items():
            if val is None or str(val).strip() == "":
                report.add_issue(ValidationIssue(
                    field=feat,
                    message=f"'{feat}' için değer girilmedi.",
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_SIMULATION_INPUT"
                ))
        return report
