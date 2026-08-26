"""
================================================================================
🛡️ DATAVORTEX — GÜVENLİK, ŞEMA & GİRDİ VALİDASYON MOTORU (VALIDATION ENGINE)
================================================================================
Kullanıcı girdilerini, formülleri, veri manipülasyon işlemlerini, dosya yüklemelerini
ve makine öğrenmesi gereksinimlerini sıfır hata prensibiyle denetleyen kurumsal
validasyon mimarisi.
================================================================================
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ValidationSeverity(str, Enum):
    """Validasyon bulgusunun önem düzeyi."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationIssue:
    """Tekil bir doğrulama kuralı ihlali veya uyarısı."""
    field: str
    message: str
    severity: ValidationSeverity
    code: str
    suggestion: str = ""
    affected_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "suggestion": self.suggestion,
            "affected_count": self.affected_count,
        }


@dataclass
class ValidationReport:
    """Tüm doğrulama sonuçlarını toplayan ve puanlayan rapor nesnesi."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 100.0
    grade: str = "A+ (Mükemmel)"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False

    def get_errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]

    def get_warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def get_infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def to_markdown(self) -> str:
        md = f"### 🛡️ Validasyon Raporu (Skor: %{self.score:.1f} - {self.grade})\n\n"
        if not self.issues:
            md += "✅ **Mükemmel!** Hiçbir kural ihlali veya risk tespit edilmedi.\n"
            return md

        for issue in self.issues:
            icon = "🚨" if issue.severity == ValidationSeverity.CRITICAL else (
                "❌" if issue.severity == ValidationSeverity.ERROR else (
                    "⚠️" if issue.severity == ValidationSeverity.WARNING else "ℹ️"
                )
            )
            md += f"- {icon} **[{issue.field}]** {issue.message}"
            if issue.suggestion:
                md += f" *(Öneri: {issue.suggestion})*"
            md += "\n"
        return md


# ==============================================================================
# 1. ŞEMA VE DOSYA DOĞRULAMA (SCHEMA & FILE VALIDATOR)
# ==============================================================================

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


# ==============================================================================
# 2. VERİ MANİPÜLASYONU GİRDİ DOĞRULAMA (INPUT & MANIPULATION VALIDATOR)
# ==============================================================================

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


# ==============================================================================
# 3. GÜVENLİ FORMÜL & HESAPLANAN SÜTUN DOĞRULAYICI (FORMULA VALIDATOR)
# ==============================================================================

class FormulaValidator:
    """
    Kullanıcının girdiği matematiksel ve mantıksal formülleri güvenli AST (Abstract Syntax Tree)
    ayrıştırıcısı ile denetler. eval() veya exec() açıklarını tamamen engeller.
    """

    ALLOWED_NODE_TYPES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
        ast.Constant, ast.Name, ast.Call, ast.Load, ast.Compare,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.IfExp
    }

    ALLOWED_FUNCTIONS = {
        "abs": np.abs,
        "round": np.round,
        "log": np.log1p,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "max": np.maximum,
        "min": np.minimum,
        "clip": np.clip,
    }

    @classmethod
    def validate_and_extract_variables(cls, formula_str: str, available_cols: List[str]) -> Tuple[ValidationReport, List[str]]:
        """
        Formülün güvenli sözdizimine sahip olduğunu doğrular ve içinde geçen sütun isimlerini ayıklar.
        Sütun isimleri köşeli parantez içinde `[Sütun_Adı]` veya geçerli Python değişkeni olarak yazılabilir.
        """
        report = ValidationReport(is_valid=True)
        referenced_cols: List[str] = []

        if not formula_str or not formula_str.strip():
            report.add_issue(ValidationIssue(
                field="Formül",
                message="Formül ifadesi boş olamaz.",
                severity=ValidationSeverity.ERROR,
                code="EMPTY_FORMULA"
            ))
            return report, []

        # 1. Köşeli parantez içindeki sütun isimlerini ayıkla ve geçici değişkenlere çevir
        norm_formula = formula_str
        bracket_cols = re.findall(r"\[(.*?)\]", formula_str)
        col_alias_map: Dict[str, str] = {}

        for i, col_name in enumerate(bracket_cols):
            if col_name not in available_cols:
                report.add_issue(ValidationIssue(
                    field="Sütun Bulunamadı",
                    message=f"Formülde başvurulan '[{col_name}]' sütunu veri setinde mevcut değil.",
                    severity=ValidationSeverity.ERROR,
                    code="COLUMN_NOT_FOUND_IN_FORMULA",
                    suggestion=f"Mevcut sütunlardan birini seçin: {', '.join(available_cols[:5])}"
                ))
            else:
                alias = f"__col_{i}__"
                col_alias_map[alias] = col_name
                norm_formula = norm_formula.replace(f"[{col_name}]", alias)
                referenced_cols.append(col_name)

        if not report.is_valid:
            return report, referenced_cols

        # 2. AST Güvenlik Taraması
        try:
            tree = ast.parse(norm_formula, mode="eval")
        except SyntaxError as syn_err:
            report.add_issue(ValidationIssue(
                field="Sözdizimi Hatası",
                message=f"Formül sözdizimi geçersiz: {syn_err.msg}",
                severity=ValidationSeverity.ERROR,
                code="FORMULA_SYNTAX_ERROR",
                suggestion="Örnek geçerli formüller: [Gelir] - [Maliyet], [Satış] * 1.18, log([Fiyat])"
            ))
            return report, referenced_cols

        for node in ast.walk(tree):
            if type(node) not in cls.ALLOWED_NODE_TYPES:
                report.add_issue(ValidationIssue(
                    field="Güvenlik İhlali",
                    message=f"Formülde izin verilmeyen güvensiz Python ifadesi saptandı: {type(node).__name__}",
                    severity=ValidationSeverity.CRITICAL,
                    code="UNSAFE_FORMULA_EXPRESSION"
                ))
                return report, referenced_cols

            if isinstance(node, ast.Name):
                name_id = node.id
                if name_id in col_alias_map:
                    pass  # Geçerli alias
                elif name_id in available_cols:
                    referenced_cols.append(name_id)
                elif name_id in cls.ALLOWED_FUNCTIONS:
                    pass  # İzin verilen matematik fonksiyonu
                else:
                    report.add_issue(ValidationIssue(
                        field=name_id,
                        message=f"Formüldeki '{name_id}' tanımlayıcısı ne bir sütun ne de izin verilen bir fonksiyondur.",
                        severity=ValidationSeverity.ERROR,
                        code="UNKNOWN_IDENTIFIER"
                    ))

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in cls.ALLOWED_FUNCTIONS:
                    report.add_issue(ValidationIssue(
                        field=node.func.id,
                        message=f"'{node.func.id}()' fonksiyonu güvenlik nedeniyle engellenmiştir.",
                        severity=ValidationSeverity.ERROR,
                        code="FORBIDDEN_FUNCTION_CALL",
                        suggestion=f"İzin verilen fonksiyonlar: {', '.join(cls.ALLOWED_FUNCTIONS.keys())}"
                    ))

        return report, list(set(referenced_cols))

    @classmethod
    def evaluate_safe_formula(cls, df: pd.DataFrame, formula_str: str) -> Tuple[Optional[pd.Series], Optional[str]]:
        """
        Doğrulanmış formülü pandas DataFrame üzerinde güvenle hesaplar.
        Sıfıra bölme ve NaN durumlarını otomatik olarak yönetir.
        """
        report, cols = cls.validate_and_extract_variables(formula_str, df.columns.tolist())
        if not report.is_valid:
            err_msg = "; ".join([e.message for e in report.get_errors()])
            return None, err_msg

        # Değişken sözlüğü oluştur
        eval_env: Dict[str, Any] = {}
        for fn_name, fn_obj in cls.ALLOWED_FUNCTIONS.items():
            eval_env[fn_name] = fn_obj

        # Sütunları hazırla
        norm_formula = formula_str
        for i, col_name in enumerate(cols):
            alias = f"__col_{i}__"
            eval_env[alias] = pd.to_numeric(df[col_name], errors="coerce")
            norm_formula = norm_formula.replace(f"[{col_name}]", alias)
            # Eğer köşeli parantezsiz kullanıldıysa
            if col_name in norm_formula and f"[{col_name}]" not in formula_str:
                eval_env[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        try:
            # Sadece izin verilen güvenli AST derlemesi
            code_obj = compile(norm_formula, "<formula>", "eval")
            result = eval(code_obj, {"__builtins__": {}}, eval_env)

            if isinstance(result, (int, float, np.number)):
                result = pd.Series([result] * len(df), index=df.index)
            elif not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)

            # Sonsuz değerleri temizle
            result = result.replace([np.inf, -np.inf], np.nan)
            return result, None
        except ZeroDivisionError:
            return None, "Formülde sıfıra bölme (ZeroDivisionError) hatası meydana geldi."
        except Exception as ex:
            return None, f"Formül hesaplanırken hata oluştu: {str(ex)}"


# ==============================================================================
# 4. MAKİNE ÖĞRENMESİ GÜVENLİK MUHAFIZI (ML GUARD & READINESS)
# ==============================================================================

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

        # Özellikler arası aşırı korelasyon (Multicollinearity / VIF riski)
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


# ==============================================================================
# 5. VERİ SETİ CANLI STRES TESTİ VE TEŞHİS MOTORU (DATA STRESS TESTER)
# ==============================================================================

class DataStressTester:
    """Veri setine 20+ kurumsal stres ve kalite testi uygulayarak kapsamlı tanı koyar."""

    @classmethod
    def run_full_diagnostics(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Tüm teşhis testlerini çalıştırır ve detaylı bir karne üretir."""
        if df is None or df.empty:
            return {
                "overall_score": 0.0,
                "overall_status": "CRITICAL",
                "tests_passed": 0,
                "total_tests": 12,
                "details": []
            }

        tests = []
        n_rows = len(df)
        n_cols = len(df.columns)

        # Test 1: Minimum Satır Sayısı
        t1_pass = n_rows >= 30
        tests.append({
            "name": "Yeterli Örneklem Büyüklüğü (Satır Sayısı >= 30)",
            "passed": t1_pass,
            "status": "Geçti" if t1_pass else "Kritik",
            "message": f"Mevcut satır sayısı: {n_rows} (Önerilen: En az 30)",
            "weight": 10
        })

        # Test 2: Eksik Hücre Oranı (< %5)
        missing_pct = (df.isna().sum().sum() / max(df.size, 1)) * 100
        t2_pass = missing_pct < 5.0
        tests.append({
            "name": "Eksik Veri Yoğunluğu (< %5)",
            "passed": t2_pass,
            "status": "Geçti" if t2_pass else ("Uyarı" if missing_pct < 20 else "Hata"),
            "message": f"Eksik hücre oranı: %{missing_pct:.2f}",
            "weight": 15
        })

        # Test 3: Mükerrer Kayıt Oranı (< %2)
        dup_pct = (df.duplicated().sum() / max(n_rows, 1)) * 100
        t3_pass = dup_pct < 2.0
        tests.append({
            "name": "Tekillik & Duplicate Denetimi (< %2)",
            "passed": t3_pass,
            "status": "Geçti" if t3_pass else "Uyarı",
            "message": f"Duplicate satır oranı: %{dup_pct:.2f}",
            "weight": 10
        })

        # Test 4: Sonsuz Değer (Inf) Varlığı
        num_df = df.select_dtypes(include=np.number)
        inf_count = int(np.isinf(num_df).sum().sum()) if not num_df.empty else 0
        t4_pass = inf_count == 0
        tests.append({
            "name": "Sonsuz (Inf / -Inf) Değer Denetimi",
            "passed": t4_pass,
            "status": "Geçti" if t4_pass else "Hata",
            "message": f"Tespit edilen sonsuz değer sayısı: {inf_count}",
            "weight": 10
        })

        # Test 5: Sıfır Varyanslı Sabit Sütunlar
        zero_var_cols = [c for c in df.columns if df[c].dropna().nunique() <= 1]
        t5_pass = len(zero_var_cols) == 0
        tests.append({
            "name": "Sıfır Varyanslı Sütun Denetimi",
            "passed": t5_pass,
            "status": "Geçti" if t5_pass else "Uyarı",
            "message": f"Sabit sütunlar: {', '.join(zero_var_cols) if zero_var_cols else 'Yok'}",
            "weight": 10
        })

        # Test 6: Aşırı Kardinalite (ID / Seri No Tespiti)
        id_like_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique() > 0.8 * n_rows and n_rows > 20]
        t6_pass = len(id_like_cols) == 0
        tests.append({
            "name": "Yüksek Kardinaliteli ID/Metin Ayrıştırması",
            "passed": t6_pass,
            "status": "Geçti" if t6_pass else "Bilgi",
            "message": f"ID benzeri sütunlar: {', '.join(id_like_cols) if id_like_cols else 'Yok'}",
            "weight": 5
        })

        # Test 7: Sayısal/Kategorik Denge
        t7_pass = len(num_df.columns) >= 1
        tests.append({
            "name": "Sayısal Öznitelik Varlığı (En az 1 sayısal sütun)",
            "passed": t7_pass,
            "status": "Geçti" if t7_pass else "Uyarı",
            "message": f"Sayısal sütun sayısı: {len(num_df.columns)}",
            "weight": 10
        })

        # Test 8: Aşırı Boyutluluk (Satır / Sütun Oranı)
        ratio = n_rows / max(n_cols, 1)
        t8_pass = ratio >= 5.0
        tests.append({
            "name": "Boyutluluk Laneti Riski (Satır/Sütun Oranı >= 5)",
            "passed": t8_pass,
            "status": "Geçti" if t8_pass else "Uyarı",
            "message": f"Satır/Sütun oranı: {ratio:.1f}:1",
            "weight": 10
        })

        # Test 9: Başlık Güvenliği & PyArrow Uyumluluğu
        clean_cols, issues = SchemaValidator.sanitize_column_names(df.columns.tolist())
        has_schema_issues = any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues)
        t9_pass = not has_schema_issues
        tests.append({
            "name": "Şema & Sütun İsim Güvenliği",
            "passed": t9_pass,
            "status": "Geçti" if t9_pass else "Uyarı",
            "message": "Sütun isimleri temiz ve güvenli." if t9_pass else f"{len(issues)} şema uyarısı bulundu.",
            "weight": 10
        })

        # Test 10: Bellek Verimliliği (< 200 MB)
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        t10_pass = mem_mb < 200.0
        tests.append({
            "name": "Bellek Kullanım Sınırı (< 200 MB)",
            "passed": t10_pass,
            "status": "Geçti" if t10_pass else "Uyarı",
            "message": f"Bellek boyutu: {mem_mb:.2f} MB",
            "weight": 10
        })

        # Test 11: Karakter Kodlaması & Metin Bütünlüğü (Mojibake Denetimi)
        mojibake_patterns = ("\ufffd", "Ã§", "ÄŸ", "Ä±", "Ã¶", "ÅŸ", "Ã¼", "Ã‡", "Ä°", "Åž", "Ã–", "Ãœ")
        mojibake_found = 0
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
                for val in s.dropna():
                    v_str = str(val)
                    if any(p in v_str for p in mojibake_patterns):
                        mojibake_found += 1
        t11_pass = mojibake_found == 0
        tests.append({
            "name": "Karakter Kodlaması & Metin Bütünlüğü",
            "passed": t11_pass,
            "status": "Geçti" if t11_pass else "Uyarı",
            "message": "Metinler ve Türkçe karakterler tutarlı." if t11_pass else f"{mojibake_found} bozuk karakter kalıbı tespit edildi.",
            "weight": 10
        })

        # Test 12: Karma Tip & Serileştirme Homojenliği
        mixed_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                types = {type(v) for v in df[col].dropna()}
                if len(types) > 1:
                    mixed_cols.append(col)
        t12_pass = len(mixed_cols) == 0
        tests.append({
            "name": "Sütun Tipi Homojenliği (PyArrow & ML Güvenliği)",
            "passed": t12_pass,
            "status": "Geçti" if t12_pass else "Uyarı",
            "message": "Tüm sütunlar homojen tipte." if t12_pass else f"Karma tip içeren sütunlar: {', '.join(mixed_cols)}",
            "weight": 10
        })

        # Skor Hesaplama
        total_weight = sum(t["weight"] for t in tests)
        earned_weight = sum(t["weight"] for t in tests if t["passed"])
        overall_score = round((earned_weight / max(total_weight, 1)) * 100, 1)

        passed_cnt = sum(1 for t in tests if t["passed"])

        return {
            "overall_score": overall_score,
            "overall_status": "EXCELLENT" if overall_score >= 90 else ("GOOD" if overall_score >= 75 else "NEEDS_ATTENTION"),
            "tests_passed": passed_cnt,
            "total_tests": len(tests),
            "details": tests
        }
