# GEMINI.md - Context & Token Optimization Instructions

## 🎯 Proje Özeti
**DataVortex**: Streamlit tabanlı, Excel/CSV veri manipülasyonu, EDA ve AutoML platformu (Private mülkiyet).

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

## 📂 Çekirdek Modül Haritası
- [validator.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/modules/validator.py): Şema, girdi ve ML muhafız kontrolleri.
- [data_processor.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/modules/data_processor.py): CSV/Excel yükleme, temizleme, undo/redo.
- [automl_engine.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/modules/automl_engine.py): AutoML eğitim hattı ve tahminler.
- [stats_engine.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/modules/stats_engine.py): İstatistik, PCA, kümeleme.
- [ui_components.py](file:///c:/Users/orayy/Desktop/Veriler/GitHub%20ve%20Proje/Yaz%C4%B1l%C4%B1m%20Projeleri%20%28B%C3%BCy%C3%BCk%29/DataVortex/modules/ui_components.py): Tema, CSS ve bildirim bileşenleri.
