# GEMINI.md - Context & Token Optimization Instructions

## 🎯 Proje Özeti
**DataVortex**: Streamlit tabanlı, Excel/CSV veri manipülasyonu, EDA ve AutoML platformu (Açık kaynak - MIT).

---

## ⚙️ Yapılandırma Parametreleri
> Bu ortamdaki tüm yapay zeka operasyonları [agent_config.yaml](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/agent_config.yaml) parametrelerine **birebir uymak zorundadır**.

| Parametre | Değer / Sınır | Açıklama |
|---|---|---|
| `execution_mode` | `strict_economy` | Minimum token, doğrudan icraat modu |
| `verbosity` | `minimal` | Fazla açıklama yapma, doğrudan sonuca odaklan |
| `max_tool_calls` | `4` araç / adım | Sonsuz araç çağırma zincirlerini engeller |
| `max_read_lines` | `120` satır | `view_file` çağrılarında üst tavan |
| `prohibit_full_reads` | `["app.py"]` | `app.py` asla tam okunamayacak |

---

## ⚡ Aşırı Düşünme & Sonsuz Döngü Muhafızı (Anti-Loop Protocol)

1. **Doğrudan İcraat (Direct Execution - Sıfır Fazla Tur)**:
   - Kullanıcı ne istediyse tahmin yürütme döngülerine girmeden doğrudan istenen dosyayı açıp düzenleyin.
   - İkinci veya üçüncü alternatif çözüm üretmeyin; ilk çalışan temiz çözümü uygulayıp durun (`stop_on_first_solution: true`).

2. **Dizin Tarama ve Keşif Yasağı**:
   - Kullanıcı bir hata veya belirli bir hedef vermedikçe tüm repoyu `list_dir` veya `grep_search` ile taramayın.
   - Yalnızca ilgili modüle bakın.

3. **Büyük Dosyalara Parçalı Okuma & Değişiklik Sınırı**:
   - `app.py` (~85KB) sadece ilgili satır aralığıyla okunur.
   - Değişikliklerde `write_to_file` ile dosya baştan yazılmaz; yalnızca `replace_file_content` kullanılır.

---

## 🚫 Yasaklı Dizin ve Dosyalar
- **Kaldırılan Testler:** `tests/`, `modules/test_runner.py`, `.pytest_cache/`, `htmlcov/`
- **İkili & Büyük Veriler:** `*.csv`, `*.xlsx`, `*.pkl`, `*.joblib`, `*.h5`, `*.parquet`, `*.onnx`

---

## 📂 Modüler Navigasyon & Görev Haritası (Task → Read Files)
> **Kural:** Herhangi bir görev için tüm depoyu taramayın. Yalnızca ilgili 1-3 dosyayı okuyun:

| Görev / Alan | İlgili Dosyalar |
|---|---|
| **Uygulama Başlangıcı & Orkestrasyon** | [app.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/app.py), [config.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/core/config.py) |
| **Session State & Bellek** | [state.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/core/state.py) |
| **Dosya Yükleme (CSV/Excel/TSV/Encoding)** | [io.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/data/io.py) |
| **Veri Temizleme & İmputasyon** | [cleaning.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/data/cleaning.py) |
| **Dönüşümler, Filtre & Formüller** | [transforms.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/data/transforms.py) |
| **Geri Al / İleri Al (Undo / Redo)** | [history.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/data/history.py) |
| **Doğrulama & Şema (AST/Input/Stress)** | [datavortex/validation/](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/validation/) (`models.py`, `schema.py`, `formula.py`, `ml_guard.py`) |
| **AutoML Eğitim Hattı** | [automl.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/ml/automl.py) |
| **Canlı & Toplu Tahmin** | [prediction.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/ml/prediction.py) |
| **Model Açıklanabilirlik & Metrikler** | [explainability.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/ml/explainability.py) |
| **İstatistik, Hipotez, K-Means, PCA** | [statistics.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/analytics/statistics.py) |
| **Gemini AI İçgörüleri** | [ai_insights.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/services/ai_insights.py) |
| **Dışa Aktarma (Excel/CSV/MD)** | [export.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/services/export.py) |
| **UI Sayfaları (7 Ana Sekme)** | [datavortex/ui/pages/](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/datavortex/ui/pages/) (`explorer.py`, `cleaning.py`, `eda.py`, `charts.py`, `ml_studio.py`, `insights.py`, `export.py`) |
| **Geriye Uyumluluk Katmanı** | [modules/](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/modules/) (`validator.py`, `data_processor.py`, `automl_engine.py`, `stats_engine.py`, `ui_components.py`) |
