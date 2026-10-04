"""
DataVortex Makine Öğrenmesi (ML) Modülü ve AutoMLEngine Facade
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline

from .automl import (
    get_recommended_target_index,
    transform_features_for_ml,
    train_models,
)
from .prediction import (
    predict_single,
    predict_batch,
)
from .explainability import (
    get_model_feature_importances,
    get_classification_confusion_matrix,
    get_regression_diagnostics,
)


class AutoMLEngine:
    """Geriye dönük tam uyumluluk (backward-compatible) sağlayan AutoML facade sınıfı."""

    get_recommended_target_index = staticmethod(get_recommended_target_index)
    transform_features_for_ml = staticmethod(transform_features_for_ml)
    train_models = staticmethod(train_models)
    predict_single = staticmethod(predict_single)
    predict_batch = staticmethod(predict_batch)


__all__ = [
    "AutoMLEngine",
    "get_recommended_target_index",
    "transform_features_for_ml",
    "train_models",
    "predict_single",
    "predict_batch",
    "get_model_feature_importances",
    "get_classification_confusion_matrix",
    "get_regression_diagnostics",
]
