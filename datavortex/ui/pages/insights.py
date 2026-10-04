"""
DataVortex — Sayfa 6: Otomatik Akıllı İçgörüler ve Gemini AI Danışmanı
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import streamlit as st

from datavortex.services.ai_insights import generate_gemini_insights
from datavortex.ui.components import NotificationHub


def render_insights_page(df: pd.DataFrame, schema_report: Any, health_dict: Dict[str, Any]) -> None:
    """Veri sağlığı risk değerlendirmesi ve Gemini AI analitik içgörülerini render eder."""
    st.markdown("### 💡 Otomatik Akıllı İçgörüler & Yönetici Özeti")
    c_ins1, c_ins2 = st.columns([1, 1])

    with c_ins1:
        st.markdown("#### 🛡️ Veri Sağlığı ve Risk Değerlendirmesi")
        mem_mb = schema_report.metrics.get("memory_mb", 0) if hasattr(schema_report, "metrics") else 0
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: {health_dict['color']}; margin-top:0;">{health_dict['grade']}</h3>
            <p><strong>Genel Sağlık Skoru:</strong> %{health_dict['score']}</p>
            <ul>
                <li><strong>Eksik Değer Oranı:</strong> %{health_dict['missing_pct']} ({health_dict['missing_cells']} hücre)</li>
                <li><strong>Duplicate Oranı:</strong> %{health_dict['dup_pct']} ({health_dict['dup_rows']} satır)</li>
                <li><strong>Toplam Boyut:</strong> {len(df)} satır x {len(df.columns)} sütun ({mem_mb} MB)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_ins2:
        st.markdown("#### 💡 Öne Çıkan Algoritmik Bulgular")
        findings = []
        num_cols_ins = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols_ins) >= 2:
            corr_mat = df[num_cols_ins].corr(numeric_only=True).abs()
            corr_arr = corr_mat.to_numpy().copy()
            np.fill_diagonal(corr_arr, 0)
            corr_mat = pd.DataFrame(corr_arr, index=corr_mat.index, columns=corr_mat.columns)
            max_corr = corr_mat.max().max() if not corr_mat.empty else 0
            if max_corr > 0.7:
                c_pair = corr_mat.stack().idxmax()
                findings.append(f"🔗 **Güçlü İlişki:** `{c_pair[0]}` ve `{c_pair[1]}` değişkenleri arasında çok yüksek korelasyon (r = {max_corr:.2f}) tespit edildi.")

        if num_cols_ins:
            cv_series = (df[num_cols_ins].std() / df[num_cols_ins].mean()).abs().dropna()
            if not cv_series.empty:
                top_var_col = cv_series.idxmax()
                findings.append(f"📊 **En Yüksek Değişkenlik:** `{top_var_col}` sütunu veri setinde en yüksek dalgalanmaya (varyasyon katsayısına) sahip metriktir.")

        cat_cols_ins = df.select_dtypes(exclude=np.number).columns.tolist()
        if cat_cols_ins:
            top_cat = cat_cols_ins[0]
            top_val = df[top_cat].mode().iloc[0] if not df[top_cat].mode().empty else "Yok"
            val_pct = (df[top_cat] == top_val).mean() * 100
            findings.append(f"🏷️ **Baskın Kategori:** `{top_cat}` alanında en sık görülen değer **'{top_val}'** (tüm verinin %{val_pct:.1f}'i).")

        if not findings:
            findings.append("Veri setinde belirgin bir kritik aykırılık veya ekstrem ilişki saptanmadı.")

        for f in findings:
            st.info(f)

    st.markdown("---")
    c_gem1, c_gem2 = st.columns([2, 1])
    with c_gem1:
        ai_api_key = st.text_input("Gemini API Key (İsteğe Bağlı)", type="password", help="Kendi Google Gemini API anahtarınızı girerek derinlemesine analiz alabilirsiniz.")
    with c_gem2:
        gemini_model_choice = st.selectbox("🤖 Model Sürümü:", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)

    if ai_api_key:
        user_question = st.text_area("Veri setiyle ilgili sormak istediğiniz soru veya rapor talebi:", placeholder="Örn: Bu verideki en karlı müşteri segmenti hangisi ve nelere dikkat edilmeli?")
        if st.button("🚀 Gemini AI ile Analiz Et", width="stretch"):
            with st.spinner(f"{gemini_model_choice} verinizi analiz ediyor..."):
                response_text, err = generate_gemini_insights(
                    api_key=ai_api_key,
                    df=df,
                    user_question=user_question,
                    model_name=gemini_model_choice
                )
                if err:
                    st.error(f"Gemini API çağrısı sırasında hata oluştu: {err}")
                else:
                    st.markdown("##### 📄 AI Yönetici Özeti & Cevap")
                    st.markdown(response_text)
                    NotificationHub.notify_success("AI analizi başarıyla tamamlandı.")
