"""
DataVortex — İnteraktif Canlı ve Toplu Tahmin Motoru (Prediction)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datavortex.ml.automl import transform_features_for_ml


def predict_single(
    best_pipe: Pipeline,
    raw_inputs: Dict[str, Any],
    engineered_features: List[str],
    is_classification: bool
) -> Tuple[Any, Optional[pd.DataFrame], Optional[str]]:
    """Canlı simülasyonda tekil tahmin ve sınıf olasılıklarını hesaplar."""
    input_df = pd.DataFrame([raw_inputs])
    input_df_trans, _ = transform_features_for_ml(input_df)

    for col_req in engineered_features:
        if col_req not in input_df_trans.columns:
            input_df_trans[col_req] = 0

    input_df_ready = input_df_trans[engineered_features]

    try:
        pred_val = best_pipe.predict(input_df_ready)[0]
        prob_df = None

        if is_classification and hasattr(best_pipe, "predict_proba"):
            probs = best_pipe.predict_proba(input_df_ready)[0]
            classes = best_pipe.classes_
            prob_df = pd.DataFrame({
                "Sınıf": classes,
                "Olasılık (%)": np.round(probs * 100, 2)
            }).sort_values(by="Olasılık (%)", ascending=False)

        return pred_val, prob_df, None
    except Exception as ex:
        return None, None, f"Tahmin hatası: {str(ex)}"


def predict_batch(
    best_pipe: Pipeline,
    target_batch_df: pd.DataFrame,
    selected_features: List[str],
    engineered_features: List[str]
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Toplu veri dosyası için tahmin üretir."""
    missing_f = [f for f in selected_features if f not in target_batch_df.columns]
    if missing_f:
        return None, f"Yüklenen dosyada modelin ihtiyaç duyduğu şu sütunlar eksik: {', '.join(missing_f)}"

    batch_ready, _ = transform_features_for_ml(target_batch_df[selected_features])
    for col_req in engineered_features:
        if col_req not in batch_ready.columns:
            batch_ready[col_req] = 0
    batch_ready = batch_ready[engineered_features]

    try:
        batch_preds = best_pipe.predict(batch_ready)
        out_df = target_batch_df.copy()
        out_df["Tahmin_Sonucu"] = batch_preds
        return out_df, None
    except Exception as ex:
        return None, f"Toplu tahmin çalıştırılırken hata oluştu: {str(ex)}"
