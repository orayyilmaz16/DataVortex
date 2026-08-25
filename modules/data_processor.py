"""
================================================================================
🧹 DATAVORTEX — GELİŞMİŞ VERİ MANİPÜLASYONU, İŞLEME & GERİ ALMA MOTORU
================================================================================
Veri temizleme, dönüştürme, aykırı değer kırpma, hesaplanan sütun türetme,
akıllı filtreleme ve işlem geçmişi (Undo/Redo) yeteneklerini sunar.
================================================================================
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .validator import SchemaValidator, InputValidator, FormulaValidator, ValidationSeverity


@dataclass
class OperationRecord:
    """Uygulanan bir veri manipülasyon işleminin kaydı."""
    action_name: str
    description: str
    timestamp: float
    affected_rows: int
    memory_before_kb: float
    memory_after_kb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "description": self.description,
            "time_str": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "affected_rows": self.affected_rows,
            "memory_delta_kb": round(self.memory_after_kb - self.memory_before_kb, 1)
        }


class DataHistoryManager:
    """Geri Alma (Undo) ve İleri Alma (Redo) için durum yöneticisi."""

    def __init__(self, max_history: int = 15):
        self.max_history = max_history
        self.undo_stack: List[Tuple[pd.DataFrame, OperationRecord]] = []
        self.redo_stack: List[Tuple[pd.DataFrame, OperationRecord]] = []

    def record_change(self, old_df: pd.DataFrame, new_df: pd.DataFrame, action_name: str, description: str):
        """Yeni bir değişikliği kaydeder ve redo yığınını sıfırlar."""
        mem_before = old_df.memory_usage(deep=True).sum() / 1024.0
        mem_after = new_df.memory_usage(deep=True).sum() / 1024.0
        affected_rows = abs(len(new_df) - len(old_df))

        record = OperationRecord(
            action_name=action_name,
            description=description,
            timestamp=time.time(),
            affected_rows=affected_rows,
            memory_before_kb=mem_before,
            memory_after_kb=mem_after
        )

        self.undo_stack.append((old_df.copy(), record))
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

        # Yeni işlem yapıldığında redo geçmişi sıfırlanır
        self.redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[OperationRecord]]:
        """Son yapılan işlemi geri alır."""
        if not self.can_undo():
            return current_df, None

        prev_df, record = self.undo_stack.pop()
        # Mevcut durumu redo yığınına kaydet
        redo_record = OperationRecord(
            action_name=f"Geri Alındı: {record.action_name}",
            description=record.description,
            timestamp=time.time(),
            affected_rows=record.affected_rows,
            memory_before_kb=current_df.memory_usage(deep=True).sum() / 1024.0,
            memory_after_kb=prev_df.memory_usage(deep=True).sum() / 1024.0
        )
        self.redo_stack.append((current_df.copy(), redo_record))
        return prev_df, record

    def redo(self, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[OperationRecord]]:
        """Geri alınan işlemi tekrar uygular."""
        if not self.can_redo():
            return current_df, None

        next_df, record = self.redo_stack.pop()
        undo_record = OperationRecord(
            action_name=f"İleri Alındı: {record.action_name}",
            description=record.description,
            timestamp=time.time(),
            affected_rows=record.affected_rows,
            memory_before_kb=current_df.memory_usage(deep=True).sum() / 1024.0,
            memory_after_kb=next_df.memory_usage(deep=True).sum() / 1024.0
        )
        self.undo_stack.append((current_df.copy(), undo_record))
        return next_df, record

    def get_history_list(self) -> List[Dict[str, Any]]:
        return [rec.to_dict() for _, rec in reversed(self.undo_stack)]


class DataProcessor:
    """Tüm veri manipülasyon işlemlerini güvenle yürüten ana işlemci."""

    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Sütun başlıklarını temizler ve benzersiz kılar.
        2. Bytes nesnelerini UTF-8 string'e çevirir.
        3. PyArrow serileştirmesi için karma tipleri homojenleştirir.
        4. Sayısal ağırlıklı sütunları float'a çevirir.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        clean_cols, _ = SchemaValidator.sanitize_column_names(df.columns.tolist())
        df.columns = clean_cols

        for col in df.columns:
            s = df[col]
            if s.dtype == "object":
                # Bytes kontrolü
                df[col] = s.apply(lambda x: x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else x)
                # Sayısal dönüşüm denemesi (%80'den fazla sayısal ise)
                try:
                    num_s = pd.to_numeric(df[col], errors="coerce")
                    valid_orig = df[col].notna().sum()
                    if valid_orig > 0 and (num_s.notna().sum() / valid_orig) >= 0.8:
                        df[col] = num_s
                except Exception:
                    pass

                # Kalan karma tipleri string yap
                if df[col].dtype == "object":
                    types = {type(x) for x in df[col].dropna()}
                    if len(types) > 1 or bytes in types:
                        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else x)

        return df

    @staticmethod
    def generate_sample_sales_data() -> pd.DataFrame:
        """Gerçekçi bir Satış, Müşteri Davranışı ve İptal/Performans Analizi örnek veri seti oluşturur."""
        np.random.seed(42)
        n = 450
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        regions = np.random.choice(["Marmara", "İç Anadolu", "Ege", "Akdeniz", "Karadeniz"], n, p=[0.35, 0.25, 0.18, 0.12, 0.10])
        categories = np.random.choice(["Elektronik", "Giyim", "Ev & Yaşam", "Kozmetik", "Spor"], n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
        units = np.random.randint(1, 45, n)
        unit_price = np.random.choice([120.0, 250.0, 480.0, 890.0, 1500.0, 2400.0, 4200.0], n)
        discount = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], n, p=[0.4, 0.2, 0.15, 0.12, 0.08, 0.05])

        revenue = np.round(units * unit_price * (1 - discount), 2)
        profit_margin = np.random.uniform(0.12, 0.42, n)
        profit = np.round(revenue * profit_margin, 2)
        customer_age = np.random.randint(19, 68, n)
        satisfaction = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.08, 0.22, 0.40, 0.25])
        payment_method = np.random.choice(["Kredi Kartı", "Havale / EFT", "Kapıda Ödeme", "Dijital Cüzdan"], n)
        loyal_customer = np.where((satisfaction >= 4) & (revenue > 2000), "Evet", "Hayır")

        df_sample = pd.DataFrame({
            "Tarih": dates,
            "Bölge": regions,
            "Kategori": categories,
            "Satış Adedi": units,
            "Birim Fiyat (TL)": unit_price,
            "İndirim Oranı": discount,
            "Toplam Gelir (TL)": revenue,
            "Net Kâr (TL)": profit,
            "Müşteri Yaşı": customer_age,
            "Müşteri Memnuniyeti": satisfaction,
            "Ödeme Yöntemi": payment_method,
            "Sadık Müşteri": loyal_customer
        })

        mask_miss = np.random.rand(*df_sample.shape) < 0.025
        df_sample = df_sample.mask(mask_miss)
        return DataProcessor.sanitize_dataframe(df_sample)

    @classmethod
    def apply_imputation(
        cls,
        df: pd.DataFrame,
        target_col: str,
        method: str,
        custom_val: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str]:
        """Eksik veri doldurma işlemini güvenle uygular."""
        new_df = df.copy()

        def _impute_series(s: pd.Series) -> pd.Series:
            if method == "Ortalama ile Doldur (Mean)" and pd.api.types.is_numeric_dtype(s):
                return s.fillna(s.mean())
            elif method == "Medyan ile Doldur (Median)" and pd.api.types.is_numeric_dtype(s):
                return s.fillna(s.median())
            elif method == "En Çok Tekrar Eden ile Doldur (Mode)":
                mode_v = s.mode().iloc[0] if not s.mode().empty else np.nan
                return s.fillna(mode_v)
            elif method == "Sabit Değer ile Doldur":
                if pd.api.types.is_numeric_dtype(s):
                    try:
                        val = float(str(custom_val).replace(",", "."))
                        return s.fillna(val)
                    except Exception:
                        return s.fillna(custom_val)
                return s.fillna(custom_val)
            elif method == "Önceki Değerle Doldur (Forward Fill)":
                return s.ffill()
            elif method == "Sonraki Değerle Doldur (Backward Fill)":
                return s.bfill()
            return s

        if target_col == "Tüm Sayısal Sütunlar":
            num_cols = new_df.select_dtypes(include=np.number).columns
            for c in num_cols:
                new_df[c] = _impute_series(new_df[c])
            msg = f"Tüm sayısal sütunlar ({len(num_cols)} adet) '{method}' yöntemiyle dolduruldu."
        elif target_col == "Tüm Kategorik Sütunlar":
            cat_cols = new_df.select_dtypes(exclude=np.number).columns
            for c in cat_cols:
                new_df[c] = _impute_series(new_df[c])
            msg = f"Tüm kategorik sütunlar ({len(cat_cols)} adet) '{method}' yöntemiyle dolduruldu."
        elif method == "Eksik İçeren Satırları Sil (Drop Rows)":
            before = len(new_df)
            new_df = new_df.dropna(subset=[target_col])
            msg = f"'{target_col}' sütununda eksik içeren {before - len(new_df)} satır silindi."
        elif method == "Eksik İçeren Sütunu Tamamen Sil (Drop Column)":
            new_df = new_df.drop(columns=[target_col])
            msg = f"'{target_col}' sütunu tablodan tamamen silindi."
        else:
            new_df[target_col] = _impute_series(new_df[target_col])
            msg = f"'{target_col}' sütunu '{method}' yöntemiyle başarıyla dolduruldu."

        return cls.sanitize_dataframe(new_df), msg

    @classmethod
    def apply_outlier_action(
        cls,
        df: pd.DataFrame,
        col: str,
        method: str,
        action: str,
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, int, str]:
        """Aykırı değer tespiti ve budama/kırpma işlemini uygular."""
        new_df = df.copy()
        series = new_df[col].dropna()

        if series.empty:
            return new_df, 0, "Seçilen sütunda geçerli sayısal veri bulunamadı."

        if "IQR" in method:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
        else:
            mean_val = series.mean()
            std_val = series.std() if series.std() > 0 else 1.0
            lower_bound = mean_val - z_threshold * std_val
            upper_bound = mean_val + z_threshold * std_val

        outlier_mask = (new_df[col] < lower_bound) | (new_df[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            return new_df, 0, f"'{col}' sütununda aykırı değer bulunamadı."

        if action == "Satırları Sil":
            new_df = new_df[~outlier_mask]
            msg = f"'{col}' sütunundaki {outlier_count} aykırı satır veri setinden çıkarıldı."
        elif action == "Sınır Değerlere Kırp (Clip)":
            new_df[col] = new_df[col].clip(lower_bound, upper_bound)
            msg = f"'{col}' sütunundaki {outlier_count} aykırı değer [{lower_bound:.2f}, {upper_bound:.2f}] aralığına kırpıldı."
        else:
            # Yeni sütun olarak etiketle
            new_df[f"{col}_Aykiri_Etiketi"] = np.where(outlier_mask, "Aykırı", "Normal")
            msg = f"'{col}' sütunu için aykırı etiket sütunu oluşturuldu ({outlier_count} aykırı)."

        return cls.sanitize_dataframe(new_df), outlier_count, msg

    @classmethod
    def apply_type_cast(cls, df: pd.DataFrame, col: str, target_type: str) -> Tuple[pd.DataFrame, str]:
        """Sütun veri tipini güvenle dönüştürür."""
        new_df = df.copy()
        if target_type == "Sayısal (Numeric / Float)":
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        elif target_type == "Tamsayı (Integer)":
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0).astype(int)
        elif target_type == "Tarih / Saat (Datetime)":
            new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
        elif target_type == "Kategorik (Category)":
            new_df[col] = new_df[col].astype("category")
        else:
            new_df[col] = new_df[col].astype(str)

        msg = f"'{col}' sütunu başarıyla '{target_type}' tipine dönüştürüldü."
        return cls.sanitize_dataframe(new_df), msg

    @classmethod
    def create_calculated_column(
        cls,
        df: pd.DataFrame,
        new_col_name: str,
        formula_str: str
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """AST tabanlı güvenli formül motoruyla yeni sütun türetir."""
        clean_name = new_col_name.strip()
        if not clean_name:
            return df, "Lütfen geçerli bir yeni sütun adı girin."

        result_series, err = FormulaValidator.evaluate_safe_formula(df, formula_str)
        if err:
            return df, err

        new_df = df.copy()
        new_df[clean_name] = result_series
        return cls.sanitize_dataframe(new_df), None

    @classmethod
    def apply_filter_rule(
        cls,
        df: pd.DataFrame,
        col: str,
        operator: str,
        val1: Any,
        val2: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, int, Optional[str]]:
        """Gelişmiş filtre kuralını uygular ve filtrelenen satır sayısını döner."""
        if col not in df.columns:
            return df, 0, f"'{col}' sütunu bulunamadı."

        s = df[col]
        mask = pd.Series([True] * len(df), index=df.index)

        try:
            if operator == "Eşittir (==)":
                if pd.api.types.is_numeric_dtype(s):
                    mask = s == float(val1)
                else:
                    mask = s.astype(str) == str(val1)
            elif operator == "Eşit Değildir (!=)":
                if pd.api.types.is_numeric_dtype(s):
                    mask = s != float(val1)
                else:
                    mask = s.astype(str) != str(val1)
            elif operator == "Büyüktür (>)":
                mask = pd.to_numeric(s, errors="coerce") > float(val1)
            elif operator == "Küçüktür (<)":
                mask = pd.to_numeric(s, errors="coerce") < float(val1)
            elif operator == "Büyük veya Eşittir (>=)":
                mask = pd.to_numeric(s, errors="coerce") >= float(val1)
            elif operator == "Küçük veya Eşittir (<=)":
                mask = pd.to_numeric(s, errors="coerce") <= float(val1)
            elif operator == "Aralığında (Between)":
                num_s = pd.to_numeric(s, errors="coerce")
                mask = (num_s >= float(val1)) & (num_s <= float(val2))
            elif operator == "İçerir (Contains)":
                mask = s.astype(str).str.contains(str(val1), case=False, na=False)
            elif operator == "İçermez (Not Contains)":
                mask = ~s.astype(str).str.contains(str(val1), case=False, na=False)
            elif operator == "Boş Olanlar (Is Null)":
                mask = s.isna()
            elif operator == "Dolu Olanlar (Is Not Null)":
                mask = s.notna()

            filtered_df = df[mask].copy()
            dropped_rows = len(df) - len(filtered_df)
            return cls.sanitize_dataframe(filtered_df), dropped_rows, None
        except Exception as ex:
            return df, 0, f"Filtreleme sırasında hata oluştu: {str(ex)}"
