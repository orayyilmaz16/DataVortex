"""
İstatistiksel Testler ve Denetimsiz Öğrenme Testleri
"""

import numpy as np
import pandas as pd
import pytest
from modules.stats_engine import StatsEngine


def test_normality_test():
    np.random.seed(42)
    normal_data = pd.Series(np.random.normal(loc=100, scale=15, size=200))
    res = StatsEngine.test_normality(normal_data)

    assert "error" not in res
    assert res["is_normal"] is True
    assert res["p_value"] > 0.05

    # Çok küçük veri seti hata vermeli
    small_data = pd.Series([1.0, 2.0])
    res_err = StatsEngine.test_normality(small_data)
    assert "error" in res_err


def test_ttest_and_anova(clean_sales_df):
    df = clean_sales_df.dropna()

    # İki grup T-Testi
    g1 = df[df["Bölge"] == "Marmara"]["Toplam Gelir (TL)"]
    g2 = df[df["Bölge"] == "Ege"]["Toplam Gelir (TL)"]
    t_res = StatsEngine.test_independent_ttest(g1, g2, "Marmara", "Ege")

    assert "error" not in t_res
    assert "t_statistic" in t_res
    assert "p_value" in t_res

    # Tek Yönlü ANOVA
    anova_res = StatsEngine.test_one_way_anova(df, "Toplam Gelir (TL)", "Bölge")
    assert "error" not in anova_res
    assert "f_statistic" in anova_res
    assert anova_res["group_count"] >= 2


def test_chi_square(clean_sales_df):
    df = clean_sales_df.dropna()
    chi_res = StatsEngine.test_chi_square(df, "Bölge", "Kategori")

    assert "error" not in chi_res
    assert "chi2_statistic" in chi_res
    assert "p_value" in chi_res
    assert "contingency_table" in chi_res


def test_kmeans_clustering(clean_sales_df):
    df = clean_sales_df.dropna()
    feats = ["Satış Adedi", "Birim Fiyat (TL)", "Toplam Gelir (TL)"]
    km_res = StatsEngine.run_kmeans_clustering(df, feats, k=3)

    assert "error" not in km_res
    assert "Küme" in km_res["result_df"].columns
    assert len(km_res["cluster_profiles"]) == 3
    assert len(km_res["inertias"]) > 0


def test_isolation_forest_anomaly(clean_sales_df):
    df = clean_sales_df.dropna()
    feats = ["Satış Adedi", "Toplam Gelir (TL)", "Net Kâr (TL)"]
    anom_res = StatsEngine.run_isolation_forest_anomaly(df, feats, contamination=0.05)

    assert "error" not in anom_res
    assert "Durum" in anom_res["anom_df"].columns
    assert "Anomali_Skoru" in anom_res["anom_df"].columns
    assert anom_res["total_anomalies"] > 0


def test_pca_decomposition(clean_sales_df):
    df = clean_sales_df.dropna()
    feats = ["Satış Adedi", "Birim Fiyat (TL)", "Toplam Gelir (TL)", "Net Kâr (TL)"]
    pca_res = StatsEngine.run_pca_decomposition(df, feats, n_components=2)

    assert "error" not in pca_res
    assert pca_res["pca_df"].shape[1] == 2
    assert len(pca_res["explained_variance_pct"]) == 2
    assert pca_res["total_explained_pct"] > 0
