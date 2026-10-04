"""
DataVortex — Reusable Görsel UI Bileşenleri ve Bildirim Merkezi
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import streamlit as st

from .theme import apply_custom_css


class UIComponents:
    """Streamlit arayüzü için kurumsal ve modern görsel bileşenler."""

    apply_custom_css = staticmethod(apply_custom_css)

    @classmethod
    def render_hero_banner(cls) -> None:
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">⚡ DataVortex | Akıllı Excel Veri Bilimi & AutoML Platformu</div>
            <p class="hero-subtitle">Kurumsal veri manipülasyonu, akıllı validasyon mimarisi, 1-tık AutoML, istatistiksel teşhis ve interaktif simülatör stüdyosu.</p>
            <div class="hero-badge">🚀 v3.5 Enterprise Edition • Sıfır Hata & Güvenlik Mimarisi Ready</div>
        </div>
        """, unsafe_allow_html=True)

    @classmethod
    def render_kpi_cards(cls, df: pd.DataFrame, health_info: Dict[str, Any]) -> None:
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
    ) -> None:
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
    ) -> None:
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
    def render_file_upload_card(cls, filename: str, metadata: Dict[str, Any]) -> None:
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
    def notify_success(message: str, icon: str = "✅") -> None:
        st.toast(f"{icon} {message}")

    @staticmethod
    def notify_error(message: str, icon: str = "❌") -> None:
        st.toast(f"{icon} {message}")

    @staticmethod
    def notify_warning(message: str, icon: str = "⚠️") -> None:
        st.toast(f"{icon} {message}")

    @staticmethod
    def notify_info(message: str, icon: str = "ℹ️") -> None:
        st.toast(f"{icon} {message}")
