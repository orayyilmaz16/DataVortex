"""
DataVortex — Data Processor & History Engine (Backward-Compatible Facade)
Bu modül geriye dönük tam uyumluluk amacıyla korunmaktadır ve
çağrıları yeni 'datavortex.data' paketine yönlendirir.
"""

from datavortex.data import (
    OperationRecord,
    DataHistoryManager,
    DataProcessor,
)

__all__ = [
    "OperationRecord",
    "DataHistoryManager",
    "DataProcessor",
]
