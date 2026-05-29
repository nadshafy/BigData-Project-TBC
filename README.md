# 🦠 TB Risk Prediction — Big Data Pipeline

> **Mata Kuliah:** IF25-40402 / Mahadata  
> **Semester:** Genap 2025/2026  
> **Institusi:** *(sesuaikan nama universitas/prodi)*

---

## 👥 Identitas Kelompok

**Kelompok 1 — Bidang Kesehatan**

| No | Nama | NIM |
|----|------|-----|
| 1  | *(Nama Anggota 1)* | *(NIM)* |
| 2  | *(Nama Anggota 2)* | *(NIM)* |
| 3  | *(Nama Anggota 3)* | *(NIM)* |
| 4  | *(Nama Anggota 4)* | *(NIM)* |
| 5  | Memory Simanjuntak | *(NIM)* |

> ✏️ *Lengkapi tabel di atas dengan nama dan NIM seluruh anggota kelompok.*

---

## 📌 Deskripsi Proyek

Proyek ini membangun sebuah **pipeline Big Data end-to-end** untuk **prediksi risiko Tuberkulosis (TBC)** di tingkat global menggunakan **Apache Spark** sebagai tulang punggung komputasi. Pipeline mengintegrasikan tiga sumber data berbeda — data beban penyakit TBC, GDP per kapita, dan kepadatan penduduk — untuk menghasilkan model prediksi berbasis machine learning serta visualisasi peta risiko interaktif.

---

## 🎯 Tujuan

- Membangun pipeline ingesti, penyimpanan, dan pemrosesan data berskala besar menggunakan Apache Spark.
- Melakukan **clustering negara** berdasarkan profil sosio-ekonomi dan epidemiologis menggunakan Bisecting K-Means.
- Melatih model **Random Forest Regressor** untuk memprediksi tingkat TBC per 100.000 penduduk.
- Mengklasifikasikan negara ke dalam zona risiko: **Rendah**, **Sedang**, dan **Tinggi**.
- Menyajikan hasil prediksi dalam bentuk **peta interaktif global** berbasis Plotly.

---

## 🗂️ Struktur Proyek

```
BigData-Project-TBC-main/
│
├── BigData.py                         # Pipeline utama Apache Spark
├── petatbc.py                         # Visualisasi peta interaktif (Plotly)
├── peta.html                          # Output peta interaktif (siap buka di browser)
├── master.csv                         # Dataset master hasil join & preprocessing
│
├── datalake/
│   ├── raw/
│   │   ├── ihme_tb/                   # Data TBC dari IHME-GBD 2023
│   │   ├── gdp/                       # Data GDP per kapita (World Bank / OWID)
│   │   └── pop_density/               # Data kepadatan penduduk (UN / OWID)
│   └── curated/
│       └── master/                    # Dataset master hasil kurated (Spark partitions)
│
├── Output/
│   ├── hasil_prediksi_tb.csv          # Hasil prediksi per negara per tahun
│   ├── hasil_evaluasi_tb.csv          # Metrik evaluasi model
│   └── hasil_evaluasi_tb.png          # Visualisasi metrik (matplotlib)
│
├── LOGBOOK_Bigdata_Kelompok1.pdf      # Logbook pengerjaan proyek
└── Laporan_Bigdata_Kelompok1.pdf      # Laporan lengkap proyek
```

---

## 🛠️ Teknologi & Library

| Teknologi | Kegunaan |
|-----------|----------|
| **Apache Spark (PySpark)** | Ingestion, Data Lake, Preprocessing, Clustering, ML, Evaluasi |
| **Spark MLlib** | Bisecting K-Means, Random Forest Regressor, Pipeline ML |
| **Pandas** | Konversi hasil untuk visualisasi & ekspor CSV akhir |
| **Matplotlib** | Visualisasi metrik evaluasi model |
| **Plotly Express** | Peta interaktif choropleth global |
| **Python 3.11** | Bahasa pemrograman utama |
| **Java 11 (JDK)** | Runtime untuk Apache Spark |
| **Hadoop** | Komponen pendukung Spark (lokal) |

---

## 📊 Sumber Dataset

| Dataset | Sumber | Deskripsi |
|---------|--------|-----------|
| **IHME-GBD 2023** | Institute for Health Metrics and Evaluation | Tingkat TBC (per 100.000 penduduk) berdasarkan negara dan tahun |
| **GDP per Capita** | World Bank via Our World in Data (OWID) | PDB per kapita sebagai prediktor sosio-ekonomi |
| **Population Density** | UN via Our World in Data (OWID) | Kepadatan penduduk sebagai prediktor demografis |

---

## ⚙️ Arsitektur Pipeline

Pipeline dibangun dalam **5 layer** berurutan:

```
Layer 1: DATA INGESTION
         └── Baca 3 sumber CSV (IHME, GDP, Population Density)

Layer 2: DATA LAKE
         └── Simpan ke zona RAW (Spark native CSV partitions)

Layer 3: PREPROCESSING
         ├── Filter metrik 'Laju' dari data IHME
         ├── Mapping nama negara (Bahasa Indonesia → Inggris)
         ├── Normalisasi kolom GDP & kepadatan penduduk
         ├── LEFT JOIN pada kunci (negara, tahun)
         ├── Imputasi nilai kosong (rata-rata per negara)
         └── Simpan ke zona CURATED

Layer 4A: CLUSTERING — Bisecting K-Means (K=3)
          └── Profil negara: Risiko Rendah / Sedang / Tinggi

Layer 4B: TRAINING — Random Forest Regressor
          └── Temporal Split: ≤2022 train | >2022 test
          └── Fitur: location, measure, year, GDP, pop_density, cluster

Layer 5: EVALUASI & OUTPUT
         ├── Metrik: R², RMSE, MAE
         ├── Klasifikasi zona risiko
         └── Export CSV + Visualisasi PNG + Peta HTML
```

---

## 📈 Hasil Evaluasi Model

Berdasarkan file `Output/hasil_evaluasi_tb.csv`:

| Metrik | Nilai |
|--------|-------|
| **Model** | Random Forest Regressor |
| **Split Year** | 2022 |
| **Training Rows** | 5.424 baris |
| **Testing Rows** | 726 baris |
| **R-Squared (R²)** | **0.9727 (97.27%)** ✅ |
| **RMSE** | 1554.2673 |
| **MAE** | 802.8105 |
| **Silhouette Score** | 0.5456 (Cluster cukup baik) |

> ✅ Model memenuhi target R² ≥ 0.80 dengan akurasi prediksi **97.27%**.

---

## 🗺️ Visualisasi

- **`peta.html`** — Peta choropleth interaktif global yang menampilkan prediksi zona risiko TBC per negara, dapat dibuka langsung di browser.
- **`Output/hasil_evaluasi_tb.png`** — Grafik metrik evaluasi model, distribusi zona risiko, dan scatter plot aktual vs prediksi.

---

## 🚀 Cara Menjalankan

### Prasyarat

- Python 3.11
- Java JDK 11
- Apache Spark (dengan Hadoop lokal)
- Library Python: `pyspark`, `pandas`, `matplotlib`, `plotly`

### Instalasi Dependensi

```bash
pip install pyspark pandas matplotlib plotly
```

### Konfigurasi Path

Sebelum menjalankan, sesuaikan path di bagian awal `BigData.py`:

```python
os.environ['JAVA_HOME']      = 'PATH_KE_JDK_KAMU'
os.environ['PYSPARK_PYTHON'] = 'PATH_KE_PYTHON_KAMU'
os.environ['HADOOP_HOME']    = 'PATH_KE_HADOOP_KAMU'

FOLDER = Path("PATH_KE_FOLDER_PROYEK_KAMU")
```

### Menjalankan Pipeline

```bash
# Jalankan pipeline utama
python BigData.py

# Jalankan visualisasi peta (setelah pipeline selesai)
python petatbc.py
```

### Output yang Dihasilkan

| File | Keterangan |
|------|------------|
| `datalake/raw/` | Data mentah tersimpan di Data Lake zona RAW |
| `datalake/curated/master/` | Data master hasil preprocessing di zona CURATED |
| `Output/hasil_prediksi_tb.csv` | Prediksi TB rate + zona risiko per negara/tahun |
| `Output/hasil_evaluasi_tb.csv` | Metrik evaluasi model (R², RMSE, MAE) |
| `Output/hasil_evaluasi_tb.png` | Visualisasi grafik evaluasi |
| `peta.html` | Peta interaktif global zona risiko TBC |

---

## 🇮🇩 Analisis Khusus: Indonesia

Pipeline menyertakan analisis khusus untuk **Indonesia**, menampilkan prediksi vs nilai aktual TB rate per tahun beserta klasifikasi zona risiko yang dihasilkan model.

---

## 📄 Dokumentasi

- 📋 **Logbook:** `LOGBOOK_Bigdata_Kelompok1.pdf` — Catatan harian proses pengerjaan proyek
- 📝 **Laporan:** `Laporan_Bigdata_Kelompok1.pdf` — Laporan lengkap metodologi, analisis, dan hasil

---

## 📜 Lisensi

Proyek ini dibuat untuk keperluan akademik pada mata kuliah **Mahadata (IF25-40402)**, Semester Genap 2025/2026. Tidak untuk digunakan secara komersial.