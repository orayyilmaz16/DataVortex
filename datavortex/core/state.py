"""
DataVortex — Session State Yönetimi ve Yardımcıları
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from datavortex.core.config import MAX_HISTORY_LENGTH


class StateKeys:
    ACTIVE_DF = "active_df"
    RAW_DF = "raw_df"
    LAST_SOURCE = "last_source"
    HISTORY_MGR = "history_manager"
    LAST_OP_FEEDBACK = "last_op_feedback"
    AUTOML_RESULT = "automl_result"
    ANOMALY_RESULT = "anom_result"


def initialize_session_state(history_manager_cls: Any) -> None:
    """Temel session state değişkenlerini başlatır."""
    if StateKeys.HISTORY_MGR not in st.session_state:
        st.session_state[StateKeys.HISTORY_MGR] = history_manager_cls(max_history=MAX_HISTORY_LENGTH)

    if StateKeys.LAST_OP_FEEDBACK not in st.session_state:
        st.session_state[StateKeys.LAST_OP_FEEDBACK] = None


def get_active_dataframe() -> Optional[pd.DataFrame]:
    """Aktif pandas DataFrame nesnesini döner."""
    return st.session_state.get(StateKeys.ACTIVE_DF)


def set_active_dataframe(df: pd.DataFrame) -> None:
    """Aktif DataFrame'i günceller."""
    st.session_state[StateKeys.ACTIVE_DF] = df


def get_automl_result() -> Optional[Dict[str, Any]]:
    """En son eğitilen AutoML sonuç sözlüğünü döner."""
    return st.session_state.get(StateKeys.AUTOML_RESULT)


def set_automl_result(res: Dict[str, Any]) -> None:
    """AutoML sonucunu kaydeder."""
    st.session_state[StateKeys.AUTOML_RESULT] = res


def clear_automl_result() -> None:
    """AutoML sonucunu temizler."""
    if StateKeys.AUTOML_RESULT in st.session_state:
        del st.session_state[StateKeys.AUTOML_RESULT]


def update_active_dataset_if_source_changed(
    raw_df: pd.DataFrame,
    source_signature: Tuple[Any, ...],
    history_manager_cls: Any
) -> None:
    """Veri kaynağı değiştiğinde aktif DataFrame'i ve geçmişi sıfırlar."""
    last_sig = st.session_state.get(StateKeys.LAST_SOURCE)
    if StateKeys.ACTIVE_DF not in st.session_state or last_sig != source_signature:
        st.session_state[StateKeys.ACTIVE_DF] = raw_df.copy()
        st.session_state[StateKeys.LAST_SOURCE] = source_signature
        st.session_state[StateKeys.HISTORY_MGR] = history_manager_cls(max_history=MAX_HISTORY_LENGTH)
        clear_automl_result()
