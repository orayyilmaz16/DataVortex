"""
DataVortex — AST Tabanlı Güvenli Formül Ayrıştırıcı ve Doğrulayıcı
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .models import ValidationIssue, ValidationReport, ValidationSeverity


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
