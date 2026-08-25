"""
================================================================================
🧪 DATAVORTEX — İSTATİSTİKSEL HİPOTEZ TESTLERİ, KÜMELEME & ANOMALİ MOTORU
================================================================================
Normallik, T-Testi, ANOVA, Ki-Kare, K-Means Kümeleme, Isolation Forest ve PCA
analizlerini ön koşul denetimleriyle yürüten istatistik motoru.
================================================================================
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class StatsEngine:
    """İstatistiksel hipotez testleri ve ileri analitik algoritmaları."""

    @staticmethod
    def test_normality(series: pd.Series) -> Dict[str, Any]:
        """Normallik testini (Shapiro-Wilk veya D'Agostino) çalıştırır."""
        s = series.dropna()
        if len(s) < 3:
            return {"error": "Normallik testi için en az 3 geçerli gözlem gereklidir."}

        if s.nunique() <= 1:
            return {"error": "Tüm değerler aynı olduğu için (sıfır varyans) normallik testi hesaplanamaz."}

        if len(s) <= 5000:
            stat_val, p_val = stats.shapiro(s)
            test_name = "Shapiro-Wilk Testi"
        else:
            stat_val, p_val = stats.normaltest(s)
            test_name = "D'Agostino's K² Testi"

        is_normal = bool(p_val > 0.05)
        return {
            "test_name": test_name,
            "statistic": float(stat_val),
            "p_value": float(p_val),
            "is_normal": is_normal,
            "sample_size": len(s),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurt())
        }

    @staticmethod
    def test_independent_ttest(
        s1: pd.Series,
        s2: pd.Series,
        label1: str = "Grup 1",
        label2: str = "Grup 2",
        equal_var: bool = False
    ) -> Dict[str, Any]:
        """İki bağımsız örneklem T-Testini çalıştırır."""
        data1 = s1.dropna()
        data2 = s2.dropna()

        if len(data1) < 2 or len(data2) < 2:
            return {"error": "Her iki grupta da en az 2 geçerli gözlem bulunmalıdır."}

        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=equal_var)
        is_significant = p_val < 0.05

        return {
            "test_name": "Welch T-Testi" if not equal_var else "Student's T-Testi",
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "is_significant": is_significant,
            "g1_label": label1,
            "g1_mean": float(data1.mean()),
            "g1_n": len(data1),
            "g2_label": label2,
            "g2_mean": float(data2.mean()),
            "g2_n": len(data2),
            "mean_diff": float(data1.mean() - data2.mean())
        }

    @staticmethod
    def test_one_way_anova(df: pd.DataFrame, num_col: str, cat_col: str) -> Dict[str, Any]:
        """Tek Yönlü ANOVA Testini (F-Testi) çalıştırır."""
        groups = []
        group_stats = []

        for name, grp in df.groupby(cat_col):
            vals = grp[num_col].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
                group_stats.append({
                    "Grup": str(name),
                    "Gözlem": len(vals),
                    "Ortalama": float(vals.mean()),
                    "Standart Sapma": float(vals.std())
                })

        if len(groups) < 2:
            return {"error": "ANOVA testi için en az 2 grupta geçerli veri bulunmalıdır."}

        f_stat, p_val = stats.f_oneway(*groups)
        return {
            "test_name": "Tek Yönlü ANOVA (F-Testi)",
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "is_significant": p_val < 0.05,
            "group_count": len(groups),
            "group_stats_df": pd.DataFrame(group_stats)
        }

    @staticmethod
    def test_chi_square(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """İki kategorik değişken arasında Ki-Kare Bağımsızlık Testi yapar."""
        contingency_table = pd.crosstab(df[col1], df[col2])
        if contingency_table.size == 0 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return {"error": "Ki-Kare testi için en az 2x2 boyutunda çapraz tablo gereklidir."}

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        low_expected_cells = np.sum(expected < 5)
        total_cells = expected.size
        low_expected_ratio = (low_expected_cells / total_cells) * 100.0

        return {
            "test_name": "Pearson Ki-Kare Bağımsızlık Testi",
            "chi2_statistic": float(chi2),
            "p_value": float(p_val),
            "dof": int(dof),
            "is_significant": p_val < 0.05,
            "contingency_table": contingency_table,
            "low_expected_ratio": round(low_expected_ratio, 1),
            "is_reliable": low_expected_ratio <= 20.0
        }

    @staticmethod
    def run_kmeans_clustering(
        df: pd.DataFrame,
        features: List[str],
        k: int = 3
    ) -> Dict[str, Any]:
        """K-Means kümeleme ve Dirsek Yöntemi (Elbow) analizi uygular."""
        c_df = df[features].dropna().copy()
        if len(c_df) < 5:
            return {"error": "Kümeleme için seçilen değişkenlerde en az 5 geçerli satır olmalıdır."}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(c_df)

        # Dirsek yöntemi
        inertias = []
        max_k_possible = min(9, len(c_df))
        k_range = list(range(2, max_k_possible))
        for current_k in k_range:
            km = KMeans(n_clusters=current_k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        # Nihai K Modeli
        actual_k = min(k, len(c_df) - 1)
        final_km = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        clusters = final_km.fit_predict(X_scaled)

        result_df = c_df.copy()
        result_df["Küme"] = [f"Küme {c + 1}" for c in clusters]
        cluster_profiles = result_df.groupby("Küme")[features].mean().reset_index()

        return {
            "result_df": result_df,
            "cluster_profiles": cluster_profiles,
            "k_range": k_range,
            "inertias": inertias,
            "selected_k": actual_k
        }

    @staticmethod
    def run_isolation_forest_anomaly(
        df: pd.DataFrame,
        features: List[str],
        contamination: float = 0.05
    ) -> Dict[str, Any]:
        """Isolation Forest algoritması ile çok boyutlu anomali tespiti yapar."""
        anom_df = df[features].dropna().copy()
        if len(anom_df) < 10:
            return {"error": "Anomali tespiti için en az 10 geçerli satır gereklidir."}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(anom_df)

        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        anom_df["Durum"] = np.where(preds == -1, "🚨 Anomali (Aykırı)", "✅ Normal")
        anom_df["Anomali_Skoru"] = np.round(scores, 4)

        total_anom = int((preds == -1).sum())
        anom_ratio = (total_anom / len(anom_df)) * 100.0

        return {
            "anom_df": anom_df,
            "total_anomalies": total_anom,
            "anomaly_ratio": round(anom_ratio, 2),
            "features": features
        }

    @staticmethod
    def run_pca_decomposition(
        df: pd.DataFrame,
        features: List[str],
        n_components: int = 3
    ) -> Dict[str, Any]:
        """Temel Bileşen Analizi (PCA) ile boyut indirgeme uygular."""
        pca_data = df[features].dropna().copy()
        if len(pca_data) < 3:
            return {"error": "PCA için en az 3 geçerli satır gereklidir."}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_data)

        n_comp = min(n_components, len(features), len(pca_data))
        pca = PCA(n_components=n_comp)
        components = pca.fit_transform(X_scaled)

        comp_cols = [f"PC{i + 1}" for i in range(n_comp)]
        pca_df = pd.DataFrame(components, columns=comp_cols, index=pca_data.index)
        explained_variance_pct = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

        return {
            "pca_df": pca_df,
            "explained_variance_pct": explained_variance_pct,
            "total_explained_pct": round(sum(explained_variance_pct), 2),
            "component_names": comp_cols
        }
