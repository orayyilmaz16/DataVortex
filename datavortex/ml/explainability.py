"""
DataVortex — Model Açıklanabilirliği ve Teşhis Metrikleri (Explainability)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline


def get_model_feature_importances(
    automl_result: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Şampiyon modelin özellik önem düzeylerini (Feature Importance veya Permutation Importance) hesaplar.
    UI'ya doğrudan grafik çizdirilebilir temiz bir DataFrame döner.
    """
    best_pipe: Pipeline = automl_result["best_pipe"]
    res = automl_result

    try:
        if hasattr(res["preprocessor"], "get_feature_names_out"):
            feat_names = list(res["preprocessor"].get_feature_names_out())
        else:
            feat_names = res["num_features"] + res["cat_features"]

        model_obj = best_pipe.named_steps.get("model")
        importances = None

        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
        elif hasattr(model_obj, "coef_"):
            importances = np.abs(model_obj.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)

        if importances is not None and len(importances) == len(feat_names):
            clean_names = [f.replace("num__", "").replace("cat__", "") for f in feat_names]
            imp_df = pd.DataFrame({
                "Özellik": clean_names,
                "Önem Düzeyi": importances
            }).sort_values(by="Önem Düzeyi", ascending=True).tail(15)
            return imp_df, None
        else:
            # Model katsayı veya feature_importances üretmiyorsa Permutation Importance kullan
            perm_res = permutation_importance(
                best_pipe, res["X_test"], res["y_test"], n_repeats=5, random_state=42
            )
            imp_df = pd.DataFrame({
                "Özellik": res["engineered_features"],
                "Önem Düzeyi": np.maximum(0, perm_res.importances_mean)
            }).sort_values(by="Önem Düzeyi", ascending=True).tail(15)
            return imp_df, None
    except Exception as ex:
        return None, str(ex)


def get_classification_confusion_matrix(
    automl_result: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sınıflandırma görevi için Confusion Matrix ve etiket dizisini hesaplar.
    Döner: (cm_array, labels, y_test_pred)
    """
    best_pipe: Pipeline = automl_result["best_pipe"]
    y_test = automl_result["y_test"]
    X_test = automl_result["X_test"]

    y_test_pred = best_pipe.predict(X_test)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    return cm, labels, y_test_pred


def get_regression_diagnostics(
    automl_result: Dict[str, Any]
) -> Tuple[pd.Series, np.ndarray, float, float]:
    """
    Regresyon görevi için gerçek vs. tahmin edilen değerleri hesaplar.
    Döner: (y_test, y_test_pred, min_val, max_val)
    """
    best_pipe: Pipeline = automl_result["best_pipe"]
    y_test = automl_result["y_test"]
    X_test = automl_result["X_test"]

    y_test_pred = best_pipe.predict(X_test)
    min_v = float(min(y_test.min(), y_test_pred.min()))
    max_v = float(max(y_test.max(), y_test_pred.max()))
    return y_test, y_test_pred, min_v, max_v
