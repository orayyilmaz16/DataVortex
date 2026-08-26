"""
================================================================================
🎨 DATAVORTEX — MODERN UI, YÜKLEME EKRANI & BİLDİRİM SİSTEMİ (UI HUB)
================================================================================
Glassmorphism görsel tasarım sistemi, animasyonlu yükleme ekranları, toast
bildirimleri, KPI kartları ve şampiyon model vitrinleri.
================================================================================
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st


class UIComponents:
    """Streamlit arayüzü için kurumsal ve modern görsel bileşenler."""

    CUSTOM_CSS = """
    <style>
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* Instant Smooth Sidebar */
        section[data-testid="stSidebar"] {
            background: #0B0F19 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.2rem !important;
        }

        /* Gradient Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.92) 50%, rgba(15, 23, 42, 0.98) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 24px 30px;
            margin-bottom: 25px;
            box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.18) 0%, transparent 60%);
            pointer-events: none;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFFFFF 20%, #93C5FD 70%, #A78BFA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 6px 0;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
            font-weight: 400;
            margin: 0;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(99, 102, 241, 0.2);
            color: #818CF8;
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 10px;
        }

        /* Metric KPI Grid */
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }
        .kpi-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 6px 16px -4px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }
        .kpi-title {
            color: #94A3B8;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .kpi-value {
            color: #F8FAFC;
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }
        .kpi-sub {
            color: #64748B;
            font-size: 0.75rem;
            margin-top: 4px;
        }

        /* Champion Model Card */
        .champion-card {
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.15) 0%, rgba(30, 41, 59, 0.8) 60%, rgba(15, 23, 42, 0.95) 100%);
            border: 1.5px solid rgba(234, 179, 8, 0.5);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px -6px rgba(234, 179, 8, 0.25);
        }
        .champion-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .champion-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #FDE047;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .champion-badge {
            background: #EAB308;
            color: #0F172A;
            font-weight: 800;
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 20px;
            letter-spacing: 0.5px;
        }

        /* Operation Feedback Pill */
        .op-feedback-pill {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 10px;
            padding: 10px 16px;
            margin-bottom: 15px;
            color: #E2E8F0;
            font-size: 0.9rem;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Glassmorphism Section Card */
        .glass-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Neon Glowing Vortex Loader Overlay */
        .neon-loader-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 35px 25px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.88) 100%);
            border: 1.5px solid rgba(99, 102, 241, 0.4);
            border-radius: 20px;
            box-shadow: 0 12px 35px -8px rgba(99, 102, 241, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.15);
            margin: 25px 0;
            backdrop-filter: blur(12px);
            text-align: center;
            animation: pulseGlow 2.5s infinite alternate ease-in-out;
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 8px 25px -6px rgba(99, 102, 241, 0.25); border-color: rgba(99, 102, 241, 0.3); }
            100% { box-shadow: 0 14px 40px -4px rgba(168, 85, 247, 0.45); border-color: rgba(168, 85, 247, 0.6); }
        }
        .vortex-spinner-ring {
            position: relative;
            width: 64px;
            height: 64px;
            margin-bottom: 18px;
        }
        .vortex-ring-1 {
            position: absolute;
            width: 100%;
            height: 100%;
            border: 4px solid transparent;
            border-top-color: #6366F1;
            border-right-color: #818CF8;
            border-radius: 50%;
            animation: spin 1s cubic-bezier(0.55, 0.15, 0.45, 0.85) infinite;
        }
        .vortex-ring-2 {
            position: absolute;
            top: 6px;
            left: 6px;
            width: 48px;
            height: 48px;
            border: 3px solid transparent;
            border-bottom-color: #A855F7;
            border-left-color: #C084FC;
            border-radius: 50%;
            animation: spinReverse 0.8s linear infinite;
        }
        .vortex-core {
            position: absolute;
            top: 20px;
            left: 20px;
            width: 24px;
            height: 24px;
            background: radial-gradient(circle, #38BDF8 0%, #6366F1 100%);
            border-radius: 50%;
            animation: corePulse 1.2s infinite ease-in-out;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes spinReverse {
            0% { transform: rotate(360deg); }
            100% { transform: rotate(0deg); }
        }
        @keyframes corePulse {
            0%, 100% { transform: scale(0.85); opacity: 0.8; }
            50% { transform: scale(1.15); opacity: 1; filter: drop-shadow(0 0 8px #38BDF8); }
        }
        .loader-title {
            color: #F8FAFC;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.3px;
        }
        .loader-subtitle {
            color: #94A3B8;
            font-size: 0.92rem;
            max-width: 480px;
            margin: 0 auto 10px auto;
        }
        .loader-step-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(99, 102, 241, 0.18);
            color: #A5B4FC;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        /* Custom Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.6);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(100, 116, 139, 0.5);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.8);
        }
    </style>
    """

    @classmethod
    def apply_custom_css(cls):
        st.markdown(cls.CUSTOM_CSS, unsafe_allow_html=True)

    @classmethod
    def render_hero_banner(cls):
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">⚡ DataVortex | Akıllı Excel Veri Bilimi & AutoML Platformu</div>
            <p class="hero-subtitle">Kurumsal veri manipülasyonu, akıllı validasyon mimarisi, 1-tık AutoML, istatistiksel teşhis ve interaktif simülatör stüdyosu.</p>
            <div class="hero-badge">🚀 v3.5 Enterprise Edition • Sıfır Hata & Güvenlik Mimarisi Ready</div>
        </div>
        """, unsafe_allow_html=True)

    @classmethod
    def render_kpi_cards(cls, df: pd.DataFrame, health_info: Dict[str, Any]):
        num_cols_cnt = len(df.select_dtypes(include=np.number).columns)
        cat_cols_cnt = len(df.select_dtypes(exclude=np.number).columns)

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">📊 Toplam Satır</div>
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-sub">Kayıt sayısı</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">📐 Toplam Sütun</div>
                <div class="kpi-value">{len(df.columns)}</div>
                <div class="kpi-sub">{num_cols_cnt} Sayısal • {cat_cols_cnt} Kategorik</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">⚠️ Eksik Hücre</div>
                <div class="kpi-value">{health_info.get('missing_cells', 0)}</div>
                <div class="kpi-sub">Toplamın %{health_info.get('missing_pct', 0.0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">🔁 Duplicate Satır</div>
                <div class="kpi-value">{health_info.get('dup_rows', 0)}</div>
                <div class="kpi-sub">Toplamın %{health_info.get('dup_pct', 0.0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">🛡️ Veri Sağlığı</div>
                <div class="kpi-value" style="color: {health_info.get('color', '#22C55E')};">%{health_info.get('score', 100.0)}</div>
                <div class="kpi-sub">{health_info.get('grade', 'A+')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @classmethod
    def render_champion_card(
        cls,
        best_model_name: str,
        target_col: str,
        task_type: str,
        primary_metric_label: str,
        primary_metric_val: str,
        duration_ms: float
    ):
        st.markdown(f"""
        <div class="champion-card">
            <div class="champion-header">
                <div class="champion-title">🏆 Şampiyon Model: {best_model_name}</div>
                <div class="champion-badge">EN İYİ BAŞARIM</div>
            </div>
            <div style="display: flex; gap: 30px; flex-wrap: wrap; color: #E2E8F0; font-size: 0.95rem;">
                <div>🎯 <strong>Hedef:</strong> <code>{target_col}</code></div>
                <div>🧠 <strong>Görev:</strong> {task_type}</div>
                <div>⭐ <strong>{primary_metric_label}:</strong> <span style="color:#FDE047; font-weight:700;">{primary_metric_val}</span></div>
                <div>⚡ <strong>Eğitim Süresi:</strong> {duration_ms} ms</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @classmethod
    def render_operation_feedback(
        cls,
        action_name: str,
        description: str,
        affected_rows: int = 0,
        duration_ms: float = 0.0
    ):
        st.markdown(f"""
        <div class="op-feedback-pill">
            <div>
                <strong>✨ {action_name}:</strong> {description}
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8;">
                {f'Etkilenen: {affected_rows} satır • ' if affected_rows > 0 else ''}{duration_ms:.1f} ms
            </div>
        </div>
        """, unsafe_allow_html=True)

    @classmethod
    def render_file_upload_card(cls, filename: str, metadata: Dict[str, Any]):
        """Yüklenen dosyanın teknik detaylarını (kodlama, ayraç, satır/sütun, süre) rozetlerle gösterir."""
        is_csv = filename.lower().endswith((".csv", ".tsv", ".txt"))
        file_type = "CSV Tablosu" if is_csv else "Excel Tablosu"
        rows = metadata.get("rows", 0)
        cols = metadata.get("cols", 0)
        dur = metadata.get("duration_ms", 0.0)
        enc = str(metadata.get("encoding", "utf-8")).upper()
        delim = metadata.get("delimiter", ",")
        delim_name = {",": "Virgül (,)", ";": "Noktalı Virgül (;)", "\t": "Tab (\\t)", "|": "Dikey Çizgi (|)"}.get(delim, delim)

        badges = [
            f"📄 {filename}",
            f"📊 {rows:,} Satır × {cols} Sütun",
            f"⚡ {dur:.1f} ms"
        ]
        if is_csv:
            badges.append(f"🔤 {enc}")
            badges.append(f"✂️ Ayraç: {delim_name}")

        badges_html = " ".join([f"<span class='hero-badge' style='margin: 3px; background: rgba(16, 185, 129, 0.15); color: #34D399; border-color: rgba(16, 185, 129, 0.35);'>{b}</span>" for b in badges])

        st.markdown(f"""
        <div class="glass-card" style="padding: 12px 16px; margin-bottom: 15px; border-left: 4px solid #10B981;">
            <div style="font-weight: 700; color: #34D399; font-size: 0.92rem; margin-bottom: 6px;">
                ✅ {file_type} Başarıyla Yüklendi ve Optimize Edildi
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {badges_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


class NotificationHub:
    """Modern toast bildirimleri ve operasyon uyarı yöneticisi."""

    @staticmethod
    def notify_success(message: str, icon: str = "✅"):
        st.toast(f"{icon} {message}")

    @staticmethod
    def notify_error(message: str, icon: str = "❌"):
        st.toast(f"{icon} {message}")

    @staticmethod
    def notify_warning(message: str, icon: str = "⚠️"):
        st.toast(f"{icon} {message}")

    @staticmethod
    def notify_info(message: str, icon: str = "ℹ️"):
        st.toast(f"{icon} {message}")
