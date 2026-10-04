"""
DataVortex Core Modülü
"""

from .config import (
    APP_TITLE,
    APP_ICON,
    APP_LAYOUT,
    SUPPORTED_FILE_EXTENSIONS,
    DEFAULT_THEMES,
    DELIMITER_MAP,
    ENCODING_MAP,
    DECIMAL_MAP,
    MAX_HISTORY_LENGTH,
)
from .state import (
    StateKeys,
    initialize_session_state,
    get_active_dataframe,
    set_active_dataframe,
    get_automl_result,
    set_automl_result,
    clear_automl_result,
    update_active_dataset_if_source_changed,
)
from .cache import (
    load_csv_cached,
    load_excel_cached,
    get_cached_sample_data,
)

__all__ = [
    "APP_TITLE",
    "APP_ICON",
    "APP_LAYOUT",
    "SUPPORTED_FILE_EXTENSIONS",
    "DEFAULT_THEMES",
    "DELIMITER_MAP",
    "ENCODING_MAP",
    "DECIMAL_MAP",
    "MAX_HISTORY_LENGTH",
    "StateKeys",
    "initialize_session_state",
    "get_active_dataframe",
    "set_active_dataframe",
    "get_automl_result",
    "set_automl_result",
    "clear_automl_result",
    "update_active_dataset_if_source_changed",
    "load_csv_cached",
    "load_excel_cached",
    "get_cached_sample_data",
]
