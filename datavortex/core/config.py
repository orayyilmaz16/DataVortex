"""
DataVortex — Genel Konfigürasyon ve Sabitler
"""

APP_TITLE = "DataVortex | Akıllı Excel Veri Bilimi & AutoML Platformu"
APP_ICON = "⚡"
APP_LAYOUT = "wide"

SUPPORTED_FILE_EXTENSIONS = ["xlsx", "xls", "xlsm", "csv", "tsv", "txt"]

DEFAULT_THEMES = ["plotly_dark", "plotly_white", "ggplot2", "seaborn", "simple_white"]

DELIMITER_MAP = {
    "Virgül (,)": ",",
    "Noktalı Virgül (;)": ";",
    "Tab (\\t)": "\t",
    "Dikey Çizgi (|)": "|"
}

ENCODING_MAP = {
    "UTF-8": "utf-8",
    "UTF-8-BOM": "utf-8-sig",
    "CP1254 (Türkçe Windows)": "cp1254",
    "ISO-8859-9": "iso-8859-9",
    "Latin-1": "latin-1"
}

DECIMAL_MAP = {
    "Nokta (.)": ".",
    "Virgül (,)": ","
}

MAX_HISTORY_LENGTH = 15
