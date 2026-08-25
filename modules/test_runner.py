"""
================================================================================
🧪 DATAVORTEX — SİSTEM & VERİ TEST KOŞUCUSU (TEST & DIAGNOSTIC RUNNER)
================================================================================
Arayüz içerisinden tek tıkla Pytest otomasyon testlerini ve veri seti stres
testlerini çalıştırıp raporlayan teşhis motoru.
================================================================================
"""

import io
import os
import sys
import time
from typing import Any, Dict, List
import pytest
from .validator import DataStressTester


class SystemTestRunner:
    """Birim, entegrasyon ve veri stres testlerini arayüz için yürüten sınıf."""

    @staticmethod
    def run_automated_test_suite() -> Dict[str, Any]:
        """Tüm pytest test paketini çalıştırır ve sonuçları ayrıştırır."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        capture_out = io.StringIO()
        sys.stdout = capture_out
        sys.stderr = capture_out

        start_time = time.time()
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests")

        exit_code = pytest.main([
            test_dir,
            "-v",
            "--tb=short",
            "-o", "junit_family=xunit2"
        ])

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        duration = round((time.time() - start_time), 2)

        raw_output = capture_out.getvalue()
        lines = raw_output.split("\n")

        parsed_tests = []
        passed_cnt = 0
        failed_cnt = 0

        for line in lines:
            if "::" in line and ("PASSED" in line or "FAILED" in line):
                parts = line.split(" ")
                test_path = parts[0]
                status = "PASSED" if "PASSED" in line else "FAILED"
                test_name = test_path.split("::")[-1] if "::" in test_path else test_path

                if status == "PASSED":
                    passed_cnt += 1
                else:
                    failed_cnt += 1

                parsed_tests.append({
                    "test_file": test_path.split("::")[0].split("/")[-1].split("\\")[-1],
                    "test_name": test_name,
                    "status": "✅ GEÇTİ" if status == "PASSED" else "❌ BAŞARISIZ",
                    "full_path": test_path
                })

        total_tests = passed_cnt + failed_cnt

        return {
            "exit_code": int(exit_code),
            "is_success": exit_code == 0,
            "total_tests": total_tests if total_tests > 0 else len(parsed_tests),
            "passed_tests": passed_cnt,
            "failed_tests": failed_cnt,
            "duration_sec": duration,
            "parsed_tests": parsed_tests,
            "raw_output": raw_output
        }

    @staticmethod
    def run_live_data_stress_test(df) -> Dict[str, Any]:
        """Aktif veri setine 20+ kuraldan oluşan kurumsal stres testini uygular."""
        return DataStressTester.run_full_diagnostics(df)
