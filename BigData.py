# ============================================================
# TB RISK PREDICTION PIPELINE — VS CODE VERSION (FINAL v3)
# Mata Kuliah: IF25-40402 / Mahadata
# Kelompok 1 — Bidang Kesehatan
# Semester Genap 2025/2026
# ============================================================
# Arsitektur:
# - Apache Spark : Ingestion, Data Lake, Preprocessing,
#                  Join, Imputasi, Clustering, ML, Evaluasi
# - Pandas       : HANYA untuk visualisasi & output CSV akhir
# ============================================================

import os
import csv
from itertools import chain
from pathlib import Path

# ── Environment Variables ────────────────────────────────────
os.environ['JAVA_HOME']      = 'D:\\Program Files\\Eclipse Adoptium\\jdk-11.0.31.11-hotspot'
os.environ['PYSPARK_PYTHON'] = 'C:\\Users\\Memory Simanjuntak\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'
os.environ['HADOOP_HOME']    = 'D:\\hadoop'

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, create_map, lit, avg,
    count, abs as spark_abs
)
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, VectorAssembler,
    StandardScaler, Imputer
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# INISIALISASI SPARK SESSION
# ============================================================
print("=" * 60)
print("  TB RISK PREDICTION — BIG DATA PIPELINE")
print("  Kelompok 1 | Mahadata 2025/2026")
print("=" * 60)

spark = SparkSession.builder \
    .appName("TB_Risk_Prediction_Spark_MultiDataset") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"\n✅ Apache Spark {spark.version} berhasil diinisialisasi!")

# ============================================================
# KONFIGURASI PATH
# ============================================================
FOLDER         = Path("D:/S.KOM/SEMESTER VI/Big Data/TBC_Project")
RAW_IHME       = str(FOLDER / "datalake/raw/ihme_tb").replace("\\", "/")
RAW_GDP        = str(FOLDER / "datalake/raw/gdp").replace("\\", "/")
RAW_POP        = str(FOLDER / "datalake/raw/pop_density").replace("\\", "/")
CURATED_MASTER = str(FOLDER / "datalake/curated/master").replace("\\", "/")

os.makedirs(str(FOLDER / "datalake/raw"),     exist_ok=True)
os.makedirs(str(FOLDER / "datalake/curated"), exist_ok=True)

# ============================================================
# LAYER 1: DATA INGESTION
# ============================================================
print("\n" + "=" * 60)
print("  LAYER 1: DATA INGESTION")
print("=" * 60)

ihme_files = sorted(FOLDER.glob("IHME-GBD_2023_DATA-*.csv"))
if not ihme_files:
    print("❌ File IHME tidak ditemukan!")
    spark.stop()
    exit(1)

spark_paths = [str(f).replace("\\", "/") for f in ihme_files]

df_ihme = spark.read.csv(spark_paths, header=True, inferSchema=True) \
               .dropna(subset=["val"]) \
               .cache()

df_gdp = spark.read.csv(
    str(FOLDER / "gdp-per-capita-worldbank.csv").replace("\\", "/"),
    header=True, inferSchema=True
)
df_pop = spark.read.csv(
    str(FOLDER / "population-density.csv").replace("\\", "/"),
    header=True, inferSchema=True
)

ihme_count = df_ihme.count()
gdp_count  = df_gdp.count()
pop_count  = df_pop.count()

print(f"\n📊 Hasil Ingestion:")
print(f"   IHME ({len(ihme_files)} file) : {ihme_count:,} baris")
print(f"   GDP per Capita    : {gdp_count:,} baris")
print(f"   Population Density: {pop_count:,} baris")
print(f"   Total data masuk  : {ihme_count + gdp_count + pop_count:,} baris")

# ============================================================
# LAYER 2: DATA LAKE (SPARK NATIVE CSV)
# ============================================================
print("\n" + "=" * 60)
print("  LAYER 2: DATA LAKE")
print("=" * 60)

df_ihme.write.mode("overwrite").option("header", "true").csv(RAW_IHME)
df_gdp.write.mode("overwrite").option("header", "true").csv(RAW_GDP)
df_pop.write.mode("overwrite").option("header", "true").csv(RAW_POP)

print(f"\n✅ Data tersimpan ke Data Lake zona RAW:")
print(f"   📁 datalake/raw/ihme_tb/")
print(f"   📁 datalake/raw/gdp/")
print(f"   📁 datalake/raw/pop_density/")

# Baca kembali dengan cast eksplisit
df_ihme = spark.read.option("header", "true") \
               .option("inferSchema", "true").csv(RAW_IHME) \
               .withColumn("val",  col("val").cast("double")) \
               .withColumn("year", col("year").cast("integer"))

df_gdp = spark.read.option("header", "true") \
              .option("inferSchema", "true").csv(RAW_GDP)

df_pop = spark.read.option("header", "true") \
              .option("inferSchema", "true").csv(RAW_POP)

print("✅ Data berhasil dibaca kembali dari Data Lake zona RAW!")

# ============================================================
# LAYER 3: PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("  LAYER 3: PREPROCESSING")
print("=" * 60)

# ── Filter IHME ─────────────────────────────────────────────
required_cols = ["location_name", "measure_name", "metric_name", "year", "val"]
df_ihme = df_ihme.dropna(subset=required_cols)
df_ihme = df_ihme.filter(col("metric_name") == "Laju")
print(f"\n📊 IHME setelah filter 'Laju': {df_ihme.count():,} baris")

# ── Mapping nama negara ──────────────────────────────────────
country_map = {
    "Pulau Marshall"    : "Marshall Islands",
    "Papua Nugini"      : "Papua New Guinea",
    "Amerika Serikat"   : "United States",
    "Republik Dominika" : "Dominican Republic",
    "Britania Raya"     : "United Kingdom",
    "Spanyol"           : "Spain",
    "Prancis"           : "France",
    "Jerman"            : "Germany",
    "Jepang"            : "Japan",
    "Korsel"            : "South Korea",
    "Korea Selatan"     : "South Korea",
    "Tiongkok"          : "China",
    "Arab Saudi"        : "Saudi Arabia",
    "Mesir"             : "Egypt",
    "Belanda"           : "Netherlands",
    "Swedia"            : "Sweden",
    "Norwegia"          : "Norway",
    "Finlandia"         : "Finland",
    "Rusia"             : "Russia",
    "Italia"            : "Italy",
    "Yunani"            : "Greece",
    "Turki"             : "Turkey",
    "Brasil"            : "Brazil",
    "Meksiko"           : "Mexico",
    "Afrika Selatan"    : "South Africa",
}
mapping_expr = create_map([lit(x) for x in chain(*country_map.items())])
df_ihme = df_ihme.withColumn(
    "translated_location",
    when(col("location_name").isin(list(country_map.keys())),
         mapping_expr[col("location_name")])
    .otherwise(col("location_name"))
)
print("✅ Mapping nama negara selesai.")

# ── Preprocessing GDP ────────────────────────────────────────
gdp_col = [c for c in df_gdp.columns if 'GDP' in c or 'gdp' in c.lower()]
df_gdp = df_gdp \
    .withColumnRenamed("Entity", "country") \
    .withColumnRenamed("Year", "year_gdp") \
    .withColumnRenamed(gdp_col[0], "gdp_per_capita") \
    .withColumn("gdp_per_capita", col("gdp_per_capita").cast("double")) \
    .withColumn("year_gdp", col("year_gdp").cast("integer"))
print(f"✅ GDP diproses. Kolom: {gdp_col[0]}")

# ── Preprocessing Population Density ────────────────────────
pop_col = [c for c in df_pop.columns
           if 'density' in c.lower() or
           ('population' in c.lower() and 'density' in c.lower())]
if not pop_col:
    pop_col = [df_pop.columns[-1]]
df_pop = df_pop \
    .withColumnRenamed("Entity", "country") \
    .withColumnRenamed("Year", "year_pop") \
    .withColumnRenamed(pop_col[0], "pop_density") \
    .withColumn("pop_density", col("pop_density").cast("double")) \
    .withColumn("year_pop", col("year_pop").cast("integer"))
df_pop = df_pop.filter(col("year_pop") > 0)
print(f"✅ Population Density diproses. Kolom: {pop_col[0]}")

# ── Join Multi-Dataset ───────────────────────────────────────
print("\n" + "─" * 40)
print("  JOIN MULTI-DATASET (Left Join)")
print("─" * 40)

df_master = df_ihme.join(
    df_gdp,
    (df_ihme.translated_location == df_gdp.country) &
    (df_ihme.year == df_gdp.year_gdp),
    how="left"
).join(
    df_pop,
    (df_ihme.translated_location == df_pop.country) &
    (df_ihme.year == df_pop.year_pop),
    how="left"
).select(
    col("location_name"),
    col("translated_location"),
    col("measure_name"),
    col("metric_name"),
    col("year").cast("integer"),
    col("val").cast("double"),
    col("gdp_per_capita").cast("double"),
    col("pop_density").cast("double")
)

master_count = df_master.count()
indo_count   = df_master.filter(col("translated_location") == "Indonesia").count()

print(f"\n📊 Hasil Join:")
print(f"   IHME (base)   : {df_ihme.count():,} baris")
print(f"   Master (join) : {master_count:,} baris")
print(f"   🇮🇩 Indonesia  : {indo_count} baris")

# ── Imputasi ─────────────────────────────────────────────────
print("\n" + "─" * 40)
print("  IMPUTASI MEDIAN (Spark MLlib)")
print("─" * 40)

print("\n📊 Missing values sebelum imputasi:")
df_master.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in ["val", "gdp_per_capita", "pop_density"]
]).show()

imputer = Imputer(
    inputCols=["gdp_per_capita", "pop_density"],
    outputCols=["gdp_per_capita", "pop_density"],
    strategy="median"
)
df_master = imputer.fit(df_master).transform(df_master)

print("📊 Missing values setelah imputasi:")
df_master.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in ["val", "gdp_per_capita", "pop_density"]
]).show()

# Simpan ke zona CURATED
df_master.write.mode("overwrite").option("header", "true").csv(CURATED_MASTER)

# Baca kembali dengan cast eksplisit
df_master = spark.read.option("header", "true") \
                 .option("inferSchema", "true").csv(CURATED_MASTER) \
                 .withColumn("val",            col("val").cast("double")) \
                 .withColumn("gdp_per_capita", col("gdp_per_capita").cast("double")) \
                 .withColumn("pop_density",    col("pop_density").cast("double")) \
                 .withColumn("year",           col("year").cast("integer"))

print(f"✅ Data curated tersimpan: {df_master.count():,} baris")
print(f"   📁 datalake/curated/master/")

# ============================================================
# LAYER 4A: BISECTING K-MEANS CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("  LAYER 4A: BISECTING K-MEANS CLUSTERING (K=3)")
print("=" * 60)

# Pakai 4 fitur agar clustering lebih bervariasi
cluster_assembler = VectorAssembler(
    inputCols=["gdp_per_capita", "pop_density", "val", "year"],
    outputCol="raw_cluster_features",
    handleInvalid="skip"
)
cluster_scaler = StandardScaler(
    inputCol="raw_cluster_features",
    outputCol="scaled_cluster_features",
    withMean=True, withStd=True
)
bkm = BisectingKMeans(
    featuresCol="scaled_cluster_features",
    predictionCol="country_cluster",
    k=3, seed=42,
    minDivisibleClusterSize=1.0
)
cluster_pipeline = Pipeline(stages=[cluster_assembler, cluster_scaler, bkm])

print("⏳ Training Bisecting K-Means...")
cluster_model = cluster_pipeline.fit(df_master)
df_master     = cluster_model.transform(df_master)
print("✅ Clustering selesai!")

# ── Evaluasi Silhouette Score ────────────────────────────────
n_clusters = df_master.select("country_cluster").distinct().count()
print(f"\n📊 Jumlah cluster unik: {n_clusters}")

if n_clusters > 1:
    evaluator_sil = ClusteringEvaluator(
        featuresCol="scaled_cluster_features",
        predictionCol="country_cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    silhouette = evaluator_sil.evaluate(df_master)
    print("\n" + "=" * 45)
    print("  EVALUASI CLUSTERING — SILHOUETTE SCORE")
    print("=" * 45)
    print(f"  Silhouette Score : {silhouette:.4f}")
    if silhouette >= 0.7:
        print("  Interpretasi     : ✅ Cluster sangat baik (≥0.70)")
    elif silhouette >= 0.5:
        print("  Interpretasi     : ✅ Cluster cukup baik (0.50–0.69)")
    elif silhouette >= 0.25:
        print("  Interpretasi     : ⚠️  Cluster lemah (0.25–0.49)")
    else:
        print("  Interpretasi     : ❌ Cluster tidak bermakna (<0.25)")
    print("=" * 45)
else:
    silhouette = 0.0
    print("⚠️ Hanya 1 cluster — Silhouette tidak bisa dihitung.")

# Label cluster berdasarkan urutan rata-rata TB rate
cluster_stats = df_master.groupBy("country_cluster").agg(
    avg("val").alias("avg_tb")
).orderBy("avg_tb").collect()

labels = ["Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"]
cluster_label_map = {
    row["country_cluster"]: labels[i]
    for i, row in enumerate(cluster_stats)
}

from pyspark.sql.functions import udf
label_udf = udf(lambda cid: cluster_label_map.get(cid, "Tidak Diketahui"), StringType())
df_master = df_master.withColumn("profil_negara", label_udf(col("country_cluster")))

print("\n📊 Statistik per Cluster:")
df_master.groupBy("country_cluster", "profil_negara").agg(
    avg("val").alias("avg_tb_rate"),
    avg("gdp_per_capita").alias("avg_gdp"),
    avg("pop_density").alias("avg_pop_density")
).orderBy("avg_tb_rate").show()

print("📊 Distribusi data per Cluster:")
df_master.groupBy("country_cluster", "profil_negara") \
         .count().orderBy("country_cluster").show()

# ============================================================
# LAYER 4B: TRAINING MODEL RANDOM FOREST
# ============================================================
print("\n" + "=" * 60)
print("  LAYER 4B: TRAINING MODEL RANDOM FOREST")
print("=" * 60)

split_year = int(df_master.approxQuantile("year", [0.80], 0.01)[0])
train_data = df_master.filter(col("year") <= split_year)
test_data  = df_master.filter(col("year") > split_year)

train_count = train_data.count()
test_count  = test_data.count()

print(f"\n📊 Temporal Split:")
print(f"   Split Year           : {split_year}")
print(f"   Training (≤{split_year}) : {train_count:,} baris")
print(f"   Testing  (>{split_year}) : {test_count:,} baris")

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in ["location_name", "measure_name"]
]
assembler = VectorAssembler(
    inputCols=[
        "location_name_idx", "measure_name_idx",
        "year", "gdp_per_capita", "pop_density", "country_cluster"
    ],
    outputCol="features",
    handleInvalid="skip"
)
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="val",
    numTrees=50,
    maxBins=300,
    seed=42
)
pipeline_rf = Pipeline(stages=indexers + [assembler, rf])

print("\n⏳ Training Random Forest... (mohon tunggu)")
model       = pipeline_rf.fit(train_data)
print("✅ Training selesai!")

print("⏳ Prediksi pada test set...")
predictions = model.transform(test_data)
print("✅ Prediksi selesai!")

# ============================================================
# LAYER 5: EVALUASI MODEL
# ============================================================
print("\n" + "=" * 60)
print("  LAYER 5: EVALUASI MODEL")
print("=" * 60)

evaluator = RegressionEvaluator(labelCol="val", predictionCol="prediction")
r2   = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae  = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

print("\n" + "=" * 50)
print("  HASIL EVALUASI RANDOM FOREST REGRESSOR")
print("=" * 50)
print(f"  R-Squared (R²) : {r2:.4f}  ({r2*100:.2f}%)")
print(f"  RMSE           : {rmse:.4f}")
print(f"  MAE            : {mae:.4f}")
print("=" * 50)

if r2 >= 0.80:
    print(f"\n  ✅ R² = {r2:.4f} — Model memenuhi target (≥0.80)")
elif r2 >= 0.60:
    print(f"\n  ⚠️  R² = {r2:.4f} — Model cukup baik")
else:
    print(f"\n  ❌ R² = {r2:.4f} — Model perlu diperbaiki")

# ── Klasifikasi Zona Risiko ──────────────────────────────────
thresholds   = predictions.approxQuantile("prediction", [0.33, 0.66], 0.01)
batas_rendah = float(thresholds[0])
batas_sedang = float(thresholds[1])

print(f"\n📊 Batas Zona Risiko:")
print(f"   Rendah : prediksi ≤ {batas_rendah:.2f}")
print(f"   Sedang : {batas_rendah:.2f} < prediksi ≤ {batas_sedang:.2f}")
print(f"   Tinggi : prediksi > {batas_sedang:.2f}")

hasil_prediksi = predictions.withColumn(
    "cluster_risiko",
    when(col("prediction") <= batas_rendah, "Rendah")
    .when(col("prediction") <= batas_sedang, "Sedang")
    .otherwise("Tinggi")
)

print("\n📊 Distribusi Zona Risiko:")
hasil_prediksi.groupBy("cluster_risiko").count().orderBy("cluster_risiko").show()

print("\n🔴 10 Negara dengan Risiko TB Tertinggi:")
hasil_prediksi.select(
    "translated_location", "profil_negara",
    "year", "val", "prediction", "cluster_risiko"
).orderBy(col("prediction").desc()).show(10, truncate=False)

# ── Analisis Khusus Indonesia ────────────────────────────────
print("\n" + "=" * 60)
print("  ANALISIS KHUSUS: INDONESIA")
print("=" * 60)

indo_pred  = hasil_prediksi.filter(col("translated_location") == "Indonesia") \
                            .select("translated_location", "year", "measure_name",
                                    "val", "prediction", "cluster_risiko",
                                    "gdp_per_capita", "pop_density", "profil_negara") \
                            .orderBy("year")
indo_count = indo_pred.count()

if indo_count > 0:
    print(f"\n📊 Prediksi vs Aktual TB — Indonesia ({indo_count} baris):")
    indo_pred.show(truncate=False)
    risiko_indo = indo_pred.select("cluster_risiko").first()[0]
    print(f"\n🇮🇩 Indonesia → Zona Risiko: {risiko_indo}")
else:
    print("\n⚠️ Indonesia tidak masuk test set.")
    df_master.filter(col("translated_location") == "Indonesia") \
             .select("translated_location", "year", "val", "profil_negara") \
             .orderBy("year").show()

# ============================================================
# VISUALISASI — Pandas hanya untuk matplotlib
# ============================================================
print("\n" + "=" * 60)
print("  LAYER 5: VISUALISASI (matplotlib)")
print("=" * 60)

risiko_pd = hasil_prediksi.groupBy("cluster_risiko").count() \
                           .orderBy("cluster_risiko").toPandas()
sample_pd = hasil_prediksi.select("val", "prediction") \
                           .filter(col("val") > 0).limit(300).toPandas()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Hasil Evaluasi Pipeline TB Risk Prediction\nKelompok 1 — Mahadata 2025/2026',
    fontsize=13, fontweight='bold'
)

# Plot 1: Metrik Evaluasi
metrics = ['R² Score', 'RMSE', 'MAE']
values  = [r2, rmse, mae]
colors  = ['#2ecc71', '#e74c3c', '#3498db']
bars = axes[0].bar(metrics, values, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('Metrik Evaluasi Model\nRandom Forest Regressor', fontsize=11)
axes[0].set_ylabel('Nilai')
axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target R²=0.80')
axes[0].legend(fontsize=8)
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=9)

# Plot 2: Distribusi Zona Risiko
risiko_colors = {'Rendah': '#27ae60', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'}
color_list    = [risiko_colors.get(r, '#95a5a6') for r in risiko_pd['cluster_risiko']]
bars2 = axes[1].bar(risiko_pd['cluster_risiko'], risiko_pd['count'],
                    color=color_list, edgecolor='black', linewidth=0.5)
axes[1].set_title('Distribusi Zona Risiko TB\n(Berdasarkan Prediksi Model)', fontsize=11)
axes[1].set_ylabel('Jumlah Data')
for bar, val in zip(bars2, risiko_pd['count']):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 str(val), ha='center', va='bottom', fontweight='bold')

# Plot 3: Aktual vs Prediksi
if len(sample_pd) > 0:
    max_val = max(sample_pd['val'].max(), sample_pd['prediction'].max())
    axes[2].scatter(sample_pd['val'], sample_pd['prediction'],
                    alpha=0.5, color='#3498db', edgecolors='navy',
                    linewidth=0.3, s=30)
    axes[2].plot([0, max_val], [0, max_val], 'r--',
                 linewidth=1.5, label='Ideal (y=x)')
axes[2].set_title(f'Aktual vs Prediksi TB Rate\n(R²={r2:.4f})', fontsize=11)
axes[2].set_xlabel('Nilai Aktual')
axes[2].set_ylabel('Nilai Prediksi')
axes[2].legend(fontsize=9)

plt.tight_layout()
output_viz = str(FOLDER / "hasil_evaluasi_tb.png")
plt.savefig(output_viz, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Visualisasi tersimpan: {output_viz}")

# ============================================================
# OUTPUT CSV
# ============================================================
print("\n" + "=" * 60)
print("  OUTPUT CSV")
print("=" * 60)

output_prediksi = str(FOLDER / "hasil_prediksi_tb.csv")
output_evaluasi = str(FOLDER / "hasil_evaluasi_tb.csv")

hasil_prediksi.select(
    "translated_location", "profil_negara",
    "measure_name", "year",
    "gdp_per_capita", "pop_density",
    "val", "prediction", "cluster_risiko"
).toPandas().to_csv(output_prediksi, index=False, encoding="utf-8-sig")

with open(output_evaluasi, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "split_year", "training_rows", "testing_rows",
                     "r_squared", "rmse", "mae", "silhouette_score"])
    writer.writerow(["Random Forest", split_year,
                     train_count, test_count,
                     round(r2, 4), round(rmse, 4),
                     round(mae, 4), round(silhouette, 4)])

print(f"✅ Output tersimpan:")
print(f"   📄 {output_prediksi}")
print(f"   📄 {output_evaluasi}")
print(f"   📊 {output_viz}")

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print(f"""
{'='*60}
  RINGKASAN PIPELINE TB RISK PREDICTION
  Kelompok 1 — Mahadata 2025/2026
{'='*60}

📦 DATASET (Multi-Sumber)
   IHME-GBD 2023     : TB Rate per 100.000 pddk (target)
   World Bank (OWID) : GDP per Kapita (prediktor sosio-ekonomi)
   UN (OWID)         : Kepadatan Penduduk (prediktor demografis)
   Total Ingestion   : {ihme_count + gdp_count + pop_count:,} baris

🔗 JOIN       : Left Join pada kunci (negara, tahun)
               Master dataset: {master_count:,} baris

🔵 CLUSTERING : Bisecting K-Means (K=3)
               Fitur : GDP, Kepadatan, TB Rate, Tahun
               Silhouette Score = {silhouette:.4f}

🤖 MODEL      : Random Forest Regressor (50 trees)
📅 SPLIT      : Temporal (≤{split_year} train | >{split_year} test)
               Training : {train_count:,} baris
               Testing  : {test_count:,} baris

📊 EVALUASI MODEL (Apache Spark MLlib)
   R²   : {r2:.4f} ({r2*100:.2f}%)
   RMSE : {rmse:.4f}
   MAE  : {mae:.4f}

✅ Pipeline selesai dieksekusi.
{'='*60}
""")

spark.stop()