# DelaySense

Keterlambatan pengiriman dapat menurunkan kepuasan pelanggan dan merusak reputasi perusahaan. Proyek ini bertujuan untuk menganalisis faktor-faktor penyebab keterlambatan dan membangun model machine learning untuk memprediksi apakah suatu pengiriman akan terlambat atau tidak.

## Repository Outline
```
1. Folder Deployment : Folder ini berisikan hal-hal yang diperlukan dalam penguploadan huggingface
2. Folder Image : Berisikan gambar hasil dari ETL Otomatis yang dilakukan di airflow
3. FP_DelaySense_DAG.py : File DAG ini berisikan syntax yang digunakan untuk ETL otomatis
4. FP_DelaySense_data_analyst.ipynb : Notebook ini berisikan EDA (Exploratory Data Analyst) dan link tableau public
5. FP_DelaySense_data_science.ipynb : Notebook ini berisikan proses modelling dalam pembuatan machine learning
6. FP_DelaySense_dataset_DA.csv : File csv ini berisikan dataset yang digunakan dalam pembuatan dashboard
7. FP_DelaySense_dataset_DS.csv : File csv ini berisikan dataset yang sudah di transformasi di airflow dan akan digunakan untuk pembuatan model
8. FP_DelaySense_GE.ipynb : Notebook ini berisikan hasil validasi data dengan great expectation
9. FP_DelaySense_inference,ipynb : Notebook ini berisikan model inference dan link HuggingFace untuk mengakses model yang sudah di up 
```

## Problem Project
Keterlambatan pengiriman dapat berdampak signifikan terhadap kepuasan pelanggan, reputasi perusahaan, dan efisiensi operasional. Menurut [TambahPinter.com](https://tambahpinter.com/keterlambatan-pengiriman-barang-atau-distribusi-dapat-merugikan/?utm_source=chatgpt.com), pelanggan yang menerima barang terlambat cenderung kehilangan kepercayaan pada layanan yang diberikan.

# Data
Kami menggunakan Dataset yang kami dapatkan di kaggle dengan link berikut: [Dataset](https://www.kaggle.com/datasets/prachi13/customer-analytics)

