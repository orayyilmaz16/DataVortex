"""
DataVortex — Statistics & Analytics Engine (Backward-Compatible Facade)
Bu modül geriye dönük tam uyumluluk amacıyla korunmaktadır ve
çağrıları yeni 'datavortex.analytics' paketine yönlendirir.
"""

from datavortex.analytics import StatsEngine

__all__ = ["StatsEngine"]
