# 🕋 Analisis Prediktif Kepuasan Jamaah Haji

Sebuah dasbor *web* interaktif yang dibangun menggunakan Streamlit untuk menganalisis dan memprediksi faktor-faktor yang memengaruhi kepuasan jamaah selama ibadah Haji.



---

## 🚀 Gambaran Umum Proyek

Manajemen kerumunan selama ibadah Haji dan Umrah adalah salah satu tantangan logistik paling kompleks di dunia. Proyek ini bertujuan untuk menganalisis dataset manajemen kerumunan untuk mengidentifikasi faktor-faktor kunci yang memengaruhi kepuasan jamaah.

Tujuan utamanya adalah mengubah analisis data mentah (dari Jupyter Notebook) menjadi produk data yang interaktif dan dapat ditindaklanjuti.

---

## ✨ Fitur Utama Dasbor

Aplikasi ini dibagi menjadi empat halaman utama:

1.  **Ringkasan Proyek:**
    * Menampilkan *Key Performance Indicators* (KPI) utama dari dataset, seperti total catatan, rata-rata waktu antri, dan rata-rata rating keamanan.
    * Memberikan gambaran umum tentang tujuan dan metodologi proyek.

2.  **Eksplorasi Data (EDA) Interaktif:**
    * Memungkinkan pengguna untuk mem-filter data secara dinamis berdasarkan **Kelompok Usia** dan **Pengalaman Jamaah**.
    * Semua visualisasi (distribusi rating, kepadatan, dll.) akan ter-update secara otomatis berdasarkan filter yang dipilih.

3.  **Simulasi Model & Performa:**
    * **Kalkulator Prediksi Kepuasan:** Fitur *live* di mana pengguna dapat memasukkan skenario (misal: waktu antri, tingkat keamanan, kepadatan) dan mendapatkan prediksi instan (`Puas`/`Tidak Puas`) dari model XGBoost.
    * Menampilkan evaluasi performa model secara transparan, termasuk *Classification Report* dan *Confusion Matrix*.

4.  **Insight & Rekomendasi:**
    * Menampilkan 5 faktor terpenting yang digunakan model untuk mengambil keputusan (Feature Importance).
    * Menerjemahkan *insight* data menjadi **rekomendasi operasional** yang jelas dan dapat ditindaklanjuti untuk meningkatkan kepuasan jamaah.

---

## 🛠️ Tumpukan Teknologi (Tech Stack)

* **Analisis Data:** Python, Pandas, NumPy
* **Visualisasi Data:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Preprocessing), XGBoost (Modeling)
* **Web App & Deployment:** Streamlit, Streamlit Community Cloud

---

## 💡 Insight Kunci & Temuan Utama

Dari analisis dan model, kami menemukan dua faktor utama yang sangat memengaruhi kepuasan jamaah:

1.  **Persepsi Keamanan adalah #1:** `Perceived_Safety_Rating` adalah faktor paling penting. Jamaah yang *merasa* aman cenderung jauh lebih puas.
    * **Rekomendasi:** Tingkatkan *visibilitas* petugas keamanan dan pastikan pencahayaan yang memadai, bukan hanya *jumlah* petugas.

2.  **Waktu Antri adalah "Pembunuh Kepuasan":** `Queue_Time_minutes` dan `Security_Checkpoint_Wait_Time` adalah faktor negatif terkuat.
    * **Rekomendasi:** Implementasikan sistem pemantauan antrian *real-time* dan buka pos pemeriksaan tambahan secara dinamis saat kepadatan terdeteksi tinggi.

---

## 📊 Sumber Data & Keterbatasan

* **Sumber Data:** Dataset yang digunakan adalah "Hajj & Umrah Crowd Management" yang diperoleh dari [Kaggle](https://www.kaggle.com/datasets/saidakd/hajj-umrah-crowd-management-dataset).
* **Keterbatasan:** Analisis menunjukkan bahwa dataset ini kemungkinan besar bersifat **sintetis** (dibuat secara artifisial), yang ditandai dengan distribusi data yang terlalu sempurna dan tidak adanya *missing values*. Oleh karena itu, performa model pada data dunia nyata mungkin berbeda.

---

![Analisis Prediktif Kepuasan Jamaah Haji Preview](Overview.png)|
**[➡️ Kunjungi Live App di Sini!](https://yutetmurpzgpnqov6kzyp9.streamlit.app//)**
