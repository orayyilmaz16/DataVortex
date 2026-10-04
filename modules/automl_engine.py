"""
DataVortex — AutoML Engine (Backward-Compatible Facade)
Bu modül geriye dönük tam uyumluluk amacıyla korunmaktadır ve
çağrıları yeni 'datavortex.ml' paketine yönlendirir.
"""

from datavortex.ml import (
    AutoMLEngine,
    get_recommended_target_index,
    transform_features_for_ml,
    train_models,
    predict_single,
    predict_batch,
)

__all__ = [
    "AutoMLEngine",
    "get_recommended_target_index",
    "transform_features_for_ml",
    "train_models",
    "predict_single",
    "predict_batch",
]
