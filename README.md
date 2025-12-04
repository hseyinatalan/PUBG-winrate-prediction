# PUBG Kazanma Oranı Tahmini

## Proje Açıklaması & Amaç

Bu proje, bir bootcamp kapsamında — hem EDA (Keşifsel Veri Analizi), hem modelleme ve hem de deploy aşamalarını içeren end-to-end bir veri bilimi projesi olarak geliştirildi. Amaç, oyuncunun maç sonunda elde edeceği bitirme yüzdesini tahmin etmek ve bu tahmine dayalı olarak daha anlaşılır bir bitiş sırası aralığı sunmak. Böylece hem teknik hem de kullanıcıya dönük bir çözüm ortaya koyuldu.

## Problem Tanımı & Hedef Değişken

Modelin tahmin ettiği ana değer, oyuncunun maç sonu bitirme yüzdesi olan winPlacePerc.
Ayrıca bu yüzdelik değer, maçtaki toplam oyuncu sayısına göre hesaplanan bir bitiş sırası aralığına dönüştürülüyor.

## Deploy Linki

https://huggingface.co/spaces/mhuseyina/PUBG_Prediction

![Deploy Demo](./Deploy_link.gif)

## Veri Seti: PUBG Finish Placement Prediction

https://www.kaggle.com/competitions/pubg-finish-placement-prediction/data

* **Toplam örnek sayısı:** 4,431,919 satır (oyuncu-maç kombinasyonu)

* **Özellikler:** 28 sütun

  * Sayısal: `kills`, `damageDealt`, `walkDistance`, `rideDistance`, `swimDistance`, `weaponsAcquired`, vb.
  * Kategorik: `matchType`, `groupId`, `matchId`, `Id`

* **Hedef değişken:** `winPlacePerc` — maç içindeki bitiş sırası yüzdesi (0 = en son, 1 = 1. sıra)

* **Açıklama:** Her satır, bir oyuncunun bir maçtaki performans istatistiklerini içerir. Oyuncular takım halinde olabilir; takım üyeleri aynı bitiş yüzdesine sahiptir. Oyuncu davranışları ve maç sonuçlarını tahmin etmek için kullanılır.

## Notebook İçerikleri

1. **EDA** – Veri keşfi ve eksik değer analizi.
2. **Baseline Model** – Basit Linear Regression ile temel tahmin ve performans ölçümü.
3. **Feature Engineering** – Yeni özellikler türetildi (`totalDistance`, `walkRideRatio`, `damagePerKill`, `playerAggression`, vb.) ve model performansı artırıldı.
4. **Model Optimization** – LightGBM ve XGBoost modelleri için Optuna ile hiperparametre optimizasyonu yapıldı, en iyi RMSE değerleri elde edildi.
5. **Model Evaluation** – Validation set üzerinde RMSE ölçüldü ve SHAP analizi ile özniteliklerin etkisi görselleştirildi.
6. **Final Pipeline** – Optimum LightGBM modeli ile 400.000 örnek üzerinde eğitim tamamlandı, doğrulama RMSE = 0.1325, model kaydedildi (`final_lgb_pubg.pkl`) ve SHAP ile öznitelik etkileri incelendi.

##  Kullanılan Teknolojiler

* **Python 3.10** 
* **Veri İşleme & Analiz:**

  * `pandas`, `numpy` — Veri manipülasyonu ve hesaplamalar
  * `matplotlib`, `seaborn` — Görselleştirme
    
* **Makine Öğrenmesi & Modeller:**

  * `scikit-learn` — Train/test split, pipeline, preprocessing, RMSE ve R² hesaplama
  * `LightGBM`, `XGBoost` — Gradient Boosting tabanlı regresyon modelleri
    
* **Hiperparametre Optimizasyonu:**

  * `Optuna` — LightGBM ve XGBoost modelleri için otomatik hiperparametre araması
    
* **Model Kaydetme ve Yükleme:**

  * `joblib` — Modellerin ve Optuna çalışmasının kaydedilmesi
    
* **Model Açıklanabilirliği:**

  * `shap` — SHAP değerleri ile özniteliklerin modele etkisinin incelenmesi
    
* **Diğer:**

  * `warnings` — Uyarıları kapatmak için
  * `os` — Dosya ve klasör işlemleri

## Local Kurulum Adımları

### Sanal Ortam Oluşturma

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
### Gerekli Paketleri Yükleme
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### Uygulama Çalıştırma
```bash
python app.py
```

## Repo Yapısı 

```plaintext
root/
│
├── app/                 # Tahmin fonksiyonları ve arayüz ile ilgili kodlar burada
│
├── final-model/         # Eğitilmiş ve kaydedilmiş final model dosyası
│
├── notebooks/           # Tüm Jupyter notebook dosyaları (EDA, feature engineering, model optimizasyon, pipeline vb.)
│
├── LICENSE              # Lisans dosyası
│
└── README.md            # Proje açıklamalarını içeren dosya

