"""
DataVortex — Gemini AI Entegrasyonu ve Token-Efficient İçgörü Servisi
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def prepare_compact_summary(df: pd.DataFrame) -> str:
    """
    Veri setini token tasarruflu, kompakt bir özet metne dönüştürür.
    Binlerce satırı API'ye göndermek yerine istatistiksel özet üretir.
    """
    n_rows, n_cols = len(df), len(df.columns)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    lines = [
        f"Veri Boyutu: {n_rows} satır, {n_cols} sütun.",
        f"Sayısal Sütunlar ({len(num_cols)}): {', '.join(num_cols[:10])}",
        f"Kategorik Sütunlar ({len(cat_cols)}): {', '.join(cat_cols[:10])}",
    ]

    if num_cols:
        desc = df[num_cols[:8]].describe().round(2).to_dict()
        lines.append(f"Temel Sayısal Özet: {desc}")

    if cat_cols:
        top_cats = {}
        for c in cat_cols[:5]:
            top_val = df[c].mode().iloc[0] if not df[c].mode().empty else "N/A"
            top_cats[c] = f"Baskın: {top_val}"
        lines.append(f"Kategorik Özet: {top_cats}")

    return "\n".join(lines)


def generate_gemini_insights(
    api_key: str,
    df: pd.DataFrame,
    user_question: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Google Gemini API kullanarak token-optimized analitik rapor veya soru cevabı üretir.
    Döner: (response_text, error_message)
    """
    if not api_key or not api_key.strip():
        return None, "Gemini API anahtarı girilmedi."

    try:
        from google import genai

        client = genai.Client(api_key=api_key.strip())
        compact_summary = prepare_compact_summary(df)

        question = user_question.strip() if user_question and user_question.strip() else (
            "Bu veri seti için üst yöneticiye sunulacak 5 maddelik stratejik içgörü raporu oluştur."
        )

        prompt = f"""
Sen uzman bir Veri Bilimci ve Kıdemli İş Zekası Danışmanısın.
Aşağıda analiz edilen veri setinin yapılandırılmış özeti yer almaktadır:

{compact_summary}

Kullanıcı Talebi / Sorusu:
{question}

Lütfen net, profesyonel, veri odaklı ve aksiyon alınabilir bir yanıt ver.
"""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        return response.text, None
    except Exception as ex:
        return None, str(ex)
