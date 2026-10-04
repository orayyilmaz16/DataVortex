"""
DataVortex — Otomatik Makine Öğrenmesi (AutoML) Motoru
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_recommended_target_index(columns: List[str], df: pd.DataFrame) -> int:
    """Hedef değişken olmaya en uygun sütunun indeksini akıllıca belirler."""
    target_keywords = [
        "sadık", "sadik", "loyal", "churn", "target", "hedef", "durum",
        "status", "label", "sonuc", "sonuç", "class", "sinif", "sınıf",
        "memnuniyet", "satisfaction"
    ]
    for i, col in enumerate(columns):
        col_lower = col.lower()
        if any(tk in col_lower for tk in target_keywords):
            s = df[col].dropna()
            if s.nunique() <= 30:
                return i

    reg_keywords = [
        "toplam gelir", "net kâr", "net kar", "gelir", "revenue",
        "profit", "fiyat", "tutar", "maas", "maaş", "score", "skor"
    ]
    for i, col in enumerate(columns):
        col_lower = col.lower()
        if any(rk in col_lower for rk in reg_keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                return i

    for i, col in enumerate(columns):
        col_lower = col.lower()
        if any(skip in col_lower for skip in ["tarih", "date", "id", "unnamed", "sütun_"]):
            continue
        s = df[col].dropna()
        n_u = s.nunique()
        if (s.dtype == "object" or s.dtype == "category" or s.dtype == "bool") and 2 <= n_u <= 10:
            return i

    for i in range(len(columns) - 1, -1, -1):
        col_lower = columns[i].lower()
        if not any(skip in col_lower for skip in ["tarih", "date", "id", "unnamed", "sütun_"]):
            if pd.api.types.is_numeric_dtype(df[columns[i]]):
                return i

    return 0


def transform_features_for_ml(X_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    1. Tarih/Zaman sütunlarını sayısal bileşenlere (_Yil, _Ay, _Gun, _HaftaninGunu) dönüştürür.
    2. Yüksek kardinaliteli ID ve benzersiz metin sütunlarını ayıklar.
    """
    X_out = pd.DataFrame(index=X_df.index)
    dropped_info: List[str] = []

    for col in X_df.columns:
        s = X_df[col]
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        if not is_dt and s.dtype == "object":
            if "tarih" in col.lower() or "date" in col.lower():
                try:
                    dt_s = pd.to_datetime(s, errors="coerce")
                    if dt_s.notna().sum() > 0.5 * len(s):
                        is_dt = True
                        s = dt_s
                except Exception:
                    pass

        if is_dt:
            dt_series = pd.to_datetime(s, errors="coerce")
            X_out[f"{col}_Ay"] = dt_series.dt.month.fillna(1).astype(int)
            X_out[f"{col}_Gun"] = dt_series.dt.day.fillna(1).astype(int)
            X_out[f"{col}_HaftaninGunu"] = dt_series.dt.dayofweek.fillna(0).astype(int)
            continue

        is_str_like = s.dtype == "object" or str(s.dtype).startswith("str") or pd.api.types.is_string_dtype(s)
        if is_str_like and s.nunique() > 0.6 * len(s) and len(s) > 20:
            dropped_info.append(col)
            continue

        X_out[col] = s

    return X_out, dropped_info


def train_models(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    selected_features: List[str],
    test_size: float = 0.2,
    progress_cb: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """Tüm modelleri eğitir, değerlendirir ve liderlik tablosunu üretir."""
    is_classification = "Sınıflandırma" in task_type or "Classification" in task_type

    ml_data = df[selected_features + [target_col]].dropna(subset=[target_col]).copy()
    y_raw = ml_data[target_col]

    X_raw_orig = ml_data[selected_features]
    X_raw, dropped_f = transform_features_for_ml(X_raw_orig)

    num_features = X_raw.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_raw.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if num_features:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, num_features))
    if cat_features:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    stratify_opt = None
    if is_classification:
        class_counts = y_raw.value_counts()
        if class_counts.min() >= 2:
            stratify_opt = y_raw

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=42, stratify=stratify_opt
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=42
        )

    if is_classification:
        model_dict = {
            "🌲 Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "⚡ Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
            "🌿 Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
            "📈 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "🎯 K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train) - 1))),
            "🌳 Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
            "🛡️ Linear SVC": LinearSVC(max_iter=2000, random_state=42),
            "📊 Gaussian Naive Bayes": GaussianNB()
        }
    else:
        model_dict = {
            "🌲 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "⚡ Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
            "🌿 Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "📈 Ridge Regression": Ridge(alpha=1.0),
            "📐 Linear Regression": LinearRegression(),
            "🎯 K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=min(5, max(1, len(X_train) - 1))),
            "🌳 Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
            "🛡️ Linear SVR": LinearSVR(max_iter=2000, random_state=42)
        }

    results_list = []
    trained_pipelines: Dict[str, Pipeline] = {}
    total_models = len(model_dict)

    for idx, (m_name, model_inst) in enumerate(model_dict.items()):
        start_t = time.time()
        try:
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model_inst)
            ])
            pipe.fit(X_train, y_train)
            fit_duration = round((time.time() - start_t) * 1000, 1)

            y_pred = pipe.predict(X_test)
            trained_pipelines[m_name] = pipe

            if is_classification:
                acc = accuracy_score(y_test, y_pred) * 100
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100

                roc_val = None
                try:
                    if hasattr(pipe, "predict_proba"):
                        probs = pipe.predict_proba(X_test)
                        if probs.shape[1] == 2:
                            roc_val = round(roc_auc_score(y_test, probs[:, 1]) * 100, 2)
                        else:
                            roc_val = round(roc_auc_score(y_test, probs, multi_class="ovr") * 100, 2)
                except Exception:
                    pass

                results_list.append({
                    "Model": m_name,
                    "Accuracy (%)": round(acc, 2),
                    "F1-Score (%)": round(f1, 2),
                    "Precision (%)": round(prec, 2),
                    "Recall (%)": round(rec, 2),
                    "ROC-AUC (%)": f"%{roc_val}" if roc_val is not None else "N/A",
                    "Eğitim Süresi (ms)": fit_duration,
                    "_score_sort": f1
                })
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                try:
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                except Exception:
                    mape = 0.0

                results_list.append({
                    "Model": m_name,
                    "R² Skoru": round(r2, 4),
                    "RMSE (Hata)": round(rmse, 2),
                    "MAE": round(mae, 2),
                    "MAPE (%)": f"%{round(mape, 2)}",
                    "Eğitim Süresi (ms)": fit_duration,
                    "_score_sort": r2
                })
        except Exception:
            pass

        if progress_cb:
            progress_cb((idx + 1) / total_models, f"{m_name} tamamlandı...")

    if not results_list:
        raise RuntimeError("Hiçbir model başarıyla eğitilemedi.")

    leaderboard_df = pd.DataFrame(results_list).sort_values(by="_score_sort", ascending=False).reset_index(drop=True)
    best_model_name = leaderboard_df.iloc[0]["Model"]
    best_pipe = trained_pipelines[best_model_name]

    return {
        "task_type": task_type,
        "target_col": target_col,
        "selected_features": selected_features,
        "engineered_features": X_raw.columns.tolist(),
        "dropped_features": dropped_f,
        "num_features": num_features,
        "cat_features": cat_features,
        "leaderboard_df": leaderboard_df,
        "best_model_name": best_model_name,
        "best_pipe": best_pipe,
        "trained_pipelines": trained_pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor
    }
