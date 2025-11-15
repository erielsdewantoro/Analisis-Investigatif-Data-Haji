import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import streamlit as st
import warnings

# Menonaktifkan peringatan spesifik (opsional, tapi membuat app lebih bersih)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------
# PENGATURAN HALAMAN (Harus menjadi perintah st pertama)
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Kepuasan Jamaah Haji & Umrah",
    page_icon="🕋",
    layout="wide",  # Menggunakan layout 'wide' agar lebih lega
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------
# FUNGSI UNTUK MEMUAT DAN MEMPROSES DATA
# -----------------------------------------------------------------
@st.cache_data  # Cache data agar tidak di-load ulang setiap kali ada interaksi
def load_data():
    # Membaca data
    df = pd.read_csv("hajj_umrah_crowd_management_dataset.csv")
    
    # --- Ini adalah langkah preprocessing dari notebook Anda ---
    
    # 1. Mengubah 'Satisfaction_Rating' menjadi biner
    # (1, 2, 3 = 0 'Tidak Puas') | (4, 5 = 1 'Puas')
    df['satisfaction_binary'] = df['Satisfaction_Rating'].apply(lambda x: 1 if x >= 4 else 0)
    
    # 2. Encoding Fitur Kategorikal (One-Hot Encoding)
    # Memilih kolom kategorikal (sesuai notebook Anda)
    categorical_cols = [
        'Crowd_Density', 'Activity_Type', 'Weather_Conditions', 'Fatigue_Level',
        'Stress_Level', 'Health_Condition', 'Age_Group', 'Pilgrim_Experience',
        'Transport_Mode', 'Emergency_Event', 'Crowd_Morale', 'AR_Navigation_Success',
        'Nationality' # Menambahkan Nationality yang ada di notebook Anda
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. Scaling Fitur Numerik
    # Memilih kolom numerik (sesuai notebook Anda)
    numeric_cols = [
        'Movement_Speed', 'Temperature', 'Sound_Level_dB', 'Queue_Time_minutes',
        'Waiting_Time_for_Transport', 'Security_Checkpoint_Wait_Time',
        'Interaction_Frequency', 'Distance_Between_People_m', 'Time_Spent_at_Location_minutes',
        'Perceived_Safety_Rating'
    ]
    
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    # 4. Mendefinisikan Fitur (X) dan Target (y)
    # (Ini adalah gabungan dari semua kolom yang telah di-encode dan di-scale)
    
    # Mengambil semua nama kolom hasil encoding
    encoded_feature_names = [col for col in df_encoded.columns if any(cat_col in col for cat_col in categorical_cols)]
    
    # Menggabungkan fitur numerik dan kategorikal yang sudah diproses
    feature_columns = numeric_cols + encoded_feature_names
    
    X = df_encoded[feature_columns]
    y = df_encoded['satisfaction_binary']
    
    # Mengembalikan data mentah (untuk EDA) dan data yg diproses (untuk Model)
    return df, X, y

# Memuat data menggunakan fungsi
df_raw, X, y = load_data()

# -----------------------------------------------------------------
# FUNGSI UNTUK MELATIH MODEL (ATAU MEMUAT MODEL JADI)
# -----------------------------------------------------------------
@st.cache_resource # Cache model agar tidak di-train ulang
def train_model(X, y):
    # Membagi data (sesuai notebook Anda)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Menghitung scale_pos_weight untuk mengatasi ketidakseimbangan
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # Melatih model XGBoost Tuned (model terbaik Anda)
    xgb_tuned_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    xgb_tuned_model.fit(X_train, y_train)
    
    # Mengembalikan model yang sudah dilatih dan data split
    return xgb_tuned_model, X_train, X_test, y_train, y_test

# Melatih model
model, X_train, X_test, y_train, y_test = train_model(X, y)

# -----------------------------------------------------------------
# SIDEBAR (NAVIGASI)
# -----------------------------------------------------------------
st.sidebar.title("Navigasi Proyek 🕋")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ("Ringkasan Proyek", "Eksplorasi Data (EDA)", "Performa Model & Prediksi")
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Proyek ini menganalisis dataset manajemen kerumunan Haji & Umrah "
    "untuk memprediksi tingkat kepuasan jamaah."
)

# -----------------------------------------------------------------
# HALAMAN 1: RINGKASAN PROYEK
# -----------------------------------------------------------------
if page == "Ringkasan Proyek":
    st.title("Analisis Prediktif Kepuasan Jamaah Haji & Umrah")
    st.markdown("""
    Selamat datang di dasbor analisis proyek ini. Dataset ini berisi 10.000 catatan sintetis 
    terkait manajemen kerumunan selama ibadah Haji dan Umrah.

    **Tujuan Utama Proyek:**
    1.  Mengeksplorasi faktor-faktor yang memengaruhi pengalaman jamaah.
    2.  Membangun model *Machine Learning* untuk memprediksi apakah seorang jamaah akan **Puas** (Rating 4-5) atau **Tidak Puas** (Rating 1-3).
    
    **Dataset:**
    -   **Sumber:** `hajj_umrah_crowd_management_dataset.csv`
    -   **Jumlah Baris:** 10.000
    -   **Jumlah Kolom:** 30

    Gunakan navigasi di *sidebar* untuk melihat Eksplorasi Data (EDA) atau hasil performa Model.
    """)
    
    st.header("Tampilan Awal Data Mentah")
    st.dataframe(df_raw.head(10))

# -----------------------------------------------------------------
# HALAMAN 2: EKSPLORASI DATA (EDA)
# -----------------------------------------------------------------
elif page == "Eksplorasi Data (EDA)":
    st.title("Eksplorasi Data (EDA)")
    st.markdown("Di halaman ini, kita akan melihat visualisasi dari data mentah untuk memahami pola.")

    # --- Visualisasi 1: Distribusi Kepuasan ---
    st.header("1. Distribusi Tingkat Kepuasan (Target)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Satisfaction_Rating', data=df_raw, palette='viridis', hue='Satisfaction_Rating', legend=False, ax=ax)
    ax.set_title('Distribusi Tingkat Kepuasan Jamaah (1=Sangat Tidak Puas, 5=Sangat Puas)', fontsize=16)
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Jumlah Jamaah', fontsize=12)
    st.pyplot(fig)
    
    st.markdown("""
    **Observasi:** Distribusi rating sangat seimbang (hampir 2000 sampel per kelas). 
    Ini tidak biasa untuk data di dunia nyata dan menunjukkan sifat sintetis dari dataset.
    """)

    # --- Visualisasi 2: Distribusi Fitur Kategorikal ---
    st.header("2. Distribusi Fitur Kategorikal")
    
    # Memilih beberapa kolom kunci untuk ditampilkan di Streamlit
    categorical_cols_to_show = [
        'Crowd_Density', 'Activity_Type', 'Weather_Conditions',
        'Fatigue_Level', 'Stress_Level', 'Health_Condition',
        'Age_Group', 'Pilgrim_Experience'
    ]
    
    # Membuat selectbox agar user bisa memilih
    selected_col = st.selectbox("Pilih Fitur Kategorikal untuk Dilihat:", categorical_cols_to_show)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.countplot(y=selected_col, data=df_raw, order=df_raw[selected_col].value_counts().index, palette='viridis', hue=selected_col, legend=False, ax=ax)
    ax.set_title(f'Distribusi Frekuensi untuk: {selected_col}', fontsize=16)
    ax.set_xlabel('Jumlah Jamaah', fontsize=12)
    ax.set_ylabel(selected_col, fontsize=12)
    st.pyplot(fig)
    
    # --- Visualisasi 3: Heatmap Korelasi ---
    st.header("3. Heatmap Korelasi Fitur Numerik")
    
    # Memilih kolom numerik dari data mentah
    numeric_cols_raw = df_raw.select_dtypes(include=np.number).columns.tolist()
    
    # Menghapus Lat/Long agar lebih fokus pada fitur sensor/pengalaman
    numeric_cols_to_corr = [col for col in numeric_cols_raw if 'Lat' not in col and 'Long' not in col]
    
    corr = df_raw[numeric_cols_to_corr].corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, annot_kws={"size": 8})
    ax.set_title('Heatmap Korelasi Antar Fitur Numerik', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    st.markdown("""
    **Observasi:**
    -   `Perceived_Safety_Rating` memiliki korelasi positif kecil dengan `Satisfaction_Rating`.
    -   `Queue_Time_minutes` memiliki korelasi negatif kecil dengan `Satisfaction_Rating`.
    -   Secara umum, korelasi antar fitur cukup rendah, yang membuat tugas prediksi menjadi menantang.
    """)


# -----------------------------------------------------------------
# HALAMAN 3: PERFORMA MODEL & PREDIKSI
# -----------------------------------------------------------------
elif page == "Performa Model & Prediksi":
    st.title("Performa Model & Prediksi")
    st.markdown("""
    Kita menggunakan **XGBoost Classifier** yang telah di-tuning dengan `scale_pos_weight` 
    untuk mengatasi target biner yang tidak seimbang (setelah kita menggabungkan rating 1-3 vs 4-5).
    """)

    # --- Bagian 1: Performa Model ---
    st.header("1. Hasil Evaluasi Model (pada Data Uji)")
    
    # Membuat prediksi
    y_pred_xgb_tuned = model.predict(X_test)
    
    # Menampilkan Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred_xgb_tuned, target_names=['Tidak Puas (0)', 'Puas (1)'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    st.markdown("""
    **Interpretasi:**
    -   **Recall (Puas):** Model ini berhasil mengidentifikasi **39%** dari semua jamaah yang *sebenarnya* Puas.
    -   **Recall (Tidak Puas):** Model ini sangat baik (84%) dalam mengidentifikasi jamaah yang *Tidak Puas*.
    -   **F1-Score (Macro Avg):** Rata-rata F1-Score adalah 0.58, yang menunjukkan performa yang *cukup* untuk dataset yang menantang ini.
    """)
    
    # Menampilkan Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        cmap='Blues', 
        display_labels=['Tidak Puas', 'Puas'],
        ax=ax
    )
    ax.set_title('Confusion Matrix - XGBoost (Tuned)')
    st.pyplot(fig)


    # --- Bagian 2: Fitur Terpenting ---
    st.header("2. Faktor Paling Berpengaruh (Feature Importance)")
    st.markdown("""
    Apa faktor utama yang digunakan model untuk mengambil keputusan?
    """)
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_15_features = feature_importances.head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_15_features, palette='viridis', hue='feature', legend=False, ax=ax)
    ax.set_title('Top 15 Feature Importances - XGBoost Model', fontsize=16)
    ax.set_xlabel('Tingkat Kepentingan', fontsize=12)
    ax.set_ylabel('Fitur', fontsize=12)
    st.pyplot(fig)
    
    st.markdown("""
    **Observasi Kunci:**
    -   **`Perceived_Safety_Rating`** adalah faktor terpenting yang menentukan kepuasan.
    -   Fitur lain seperti `Time_Spent_at_Location_minutes`, `Queue_Time_minutes`, dan `Security_Checkpoint_Wait_Time` juga memainkan peran penting.
    -   Kepadatan (`Crowd_Density_Medium`, `Crowd_Density_Low`) juga memengaruhi hasil.
    """)
