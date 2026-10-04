"""
DataVortex — Veri Seti Stres Testi ve Tanı Motoru
"""

from typing import Any, Dict
import numpy as np
import pandas as pd

from .models import ValidationSeverity
from .schema import SchemaValidator


class DataStressTester:
    """Veri setine 12+ kurumsal stres ve kalite testi uygulayarak kapsamlı tanı koyar."""

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
