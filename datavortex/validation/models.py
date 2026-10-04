"""
DataVortex — Validasyon Veri Modelleri ve Rapor Nesneleri
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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

    def add_issue(self, issue: ValidationIssue) -> None:
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
