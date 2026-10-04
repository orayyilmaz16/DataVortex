"""
DataVortex — UI Components & Notification Hub (Backward-Compatible Facade)
Bu modül geriye dönük tam uyumluluk amacıyla korunmaktadır ve
çağrıları yeni 'datavortex.ui' paketine yönlendirir.
"""

from datavortex.ui import (
    UIComponents,
    NotificationHub,
    apply_custom_css,
    CUSTOM_CSS,
)

__all__ = [
    "UIComponents",
    "NotificationHub",
    "apply_custom_css",
    "CUSTOM_CSS",
]
