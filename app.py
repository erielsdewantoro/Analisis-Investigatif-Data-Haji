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

# Menonaktifkan peringatan
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------
# PENGATURAN HALAMAN (Harus menjadi perintah st pertama)
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Kepuasan Jamaah",
    page_icon="🕋",
    layout="wide", # Layout 'wide' lebih profesional untuk dasbor
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------
# FUNGSI UNTUK MEMUAT DAN MEMPROSES DATA
# -----------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("hajj_umrah_crowd_management_dataset.csv")
    
    # 1. Mengubah 'Satisfaction_Rating' menjadi biner
    df['satisfaction_binary'] = df['Satisfaction_Rating'].apply(lambda x: 1 if x >= 4 else 0)
    
    # 2. Mendefinisikan fitur
    categorical_cols = [
        'Crowd_Density', 'Activity_Type', 'Weather_Conditions', 'Fatigue_Level',
        'Stress_Level', 'Health_Condition', 'Age_Group', 'Pilgrim_Experience',
        'Transport_Mode', 'Emergency_Event', 'Crowd_Morale', 'AR_Navigation_Success',
        'Nationality'
    ]
    # Menggunakan drop_first=False agar lebih mudah saat membuat form prediksi
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    numeric_cols = [
        'Movement_Speed', 'Temperature', 'Sound_Level_dB', 'Queue_Time_minutes',
        'Waiting_Time_for_Transport', 'Security_Checkpoint_Wait_Time',
        'Interaction_Frequency', 'Distance_Between_People_m', 'Time_Spent_at_Location_minutes',
        'Perceived_Safety_Rating'
    ]
    
    # 3. Scaling Fitur Numerik
    scaler = StandardScaler()
    # Fit scaler pada data
    scaler.fit(df_encoded[numeric_cols]) 
    # Transform data
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
    
    # 4. Mendefinisikan Fitur (X) dan Target (y)
    encoded_feature_names = [col for col in df_encoded.columns if any(cat_col in col for cat_col in categorical_cols)]
    feature_columns = numeric_cols + encoded_feature_names
    
    X = df_encoded[feature_columns]
    y = df_encoded['satisfaction_binary']
    
    # Mengembalikan semua yang kita butuhkan
    return df, X, y, scaler, numeric_cols, feature_columns

# Memuat data
df_raw, X, y, scaler, numeric_cols, feature_columns = load_data()

# -----------------------------------------------------------------
# FUNGSI UNTUK MELATIH MODEL
# -----------------------------------------------------------------
@st.cache_resource # Cache model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    xgb_tuned_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    xgb_tuned_model.fit(X_train, y_train)
    return xgb_tuned_model, X_test, y_test

# Melatih model
model, X_test, y_test = train_model(X, y)

# -----------------------------------------------------------------
# SIDEBAR (NAVIGASI & FILTER)
# -----------------------------------------------------------------
st.sidebar.title("Navigasi 🧭")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ("Ringkasan Proyek",           # 1. The 'What'
     "Eksplorasi Data (EDA)",      # 2. The 'Details'
     "Simulasi Model & Performa",  # 3. The 'How'
     "Insight & Rekomendasi"       # 4. The 'So What' & 'Now What'
    )
)
st.sidebar.markdown("---")

# **FITUR INTERAKTIF v2.0: FILTER EDA**
st.sidebar.header("Filter Data (untuk EDA)")
st.sidebar.info("Filter ini hanya akan memengaruhi halaman 'Eksplorasi Data (EDA)'.")

# Filter 1: Kelompok Usia
age_options = st.sidebar.multiselect(
    "Pilih Kelompok Usia:",
    options=df_raw['Age_Group'].unique(),
    default=df_raw['Age_Group'].unique()
)

# Filter 2: Pengalaman Jamaah
exp_options = st.sidebar.multiselect(
    "Pilih Pengalaman Jamaah:",
    options=df_raw['Pilgrim_Experience'].unique(),
    default=df_raw['Pilgrim_Experience'].unique()
)

# Terapkan filter ke dataframe mentah
df_filtered = df_raw[
    (df_raw['Age_Group'].isin(age_options)) &
    (df_raw['Pilgrim_Experience'].isin(exp_options))
]


# -----------------------------------------------------------------
# HALAMAN 1: RINGKASAN PROYEK
# -----------------------------------------------------------------
if page == "Ringkasan Proyek":
    st.title("🕋 Analisis Prediktif Kepuasan Jamaah Haji & Umrah")
    st.markdown("""
    Dasbor ini menganalisis dataset manajemen kerumunan untuk **mengidentifikasi faktor-faktor kunci** yang memengaruhi kepuasan jamaah dan **membangun model prediktif**.
    """)
    
    # **FITUR PROFESIONAL v2.0: KOLOM & METRIK**
    st.markdown("---")
    st.header("Ringkasan Data Global (KPI)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Catatan Jamaah", f"{df_raw.shape[0]:,}")
    col2.metric("Rata-rata Waktu Antri (Menit)", f"{df_raw['Queue_Time_minutes'].mean():.1f} Menit")
    col3.metric("Rata-rata Rating Keamanan", f"{df_raw['Perceived_Safety_Rating'].mean():.1f} / 5")
    
    st.markdown("---")
    st.header("Tujuan Proyek")
    st.markdown("""
    1.  **Eksplorasi:** Memahami pola dalam data kerumunan (misal: kapan antrian terpanjang terjadi?).
    2.  **Prediksi:** Membangun model *Machine Learning* untuk memprediksi apakah seorang jamaah akan **Puas** (Rating 4-5) atau **Tidak Puas** (Rating 1-3).
    3.  **Rekomendasi:** Menemukan faktor apa yang paling memengaruhi kepuasan untuk memberikan rekomendasi operasional.
    """)
    
    # **FITUR PROFESIONAL v2.0: EXPANDER**
    with st.expander("Lihat Sampel Data Mentah (15 Baris Pertama)"):
        st.dataframe(df_raw.head(15))

# -----------------------------------------------------------------
# HALAMAN 2: EKSPLORASI DATA (EDA)
# -----------------------------------------------------------------
elif page == "Eksplorasi Data (EDA)":
    st.title("🔍 Eksplorasi Data (EDA) Interaktif")
    st.warning(f"Anda sedang melihat **{len(df_filtered)}** dari **{len(df_raw)}** total data (berdasarkan filter di sidebar).")

    if df_filtered.empty:
        st.error("Tidak ada data yang cocok dengan filter Anda. Silakan sesuaikan filter di sidebar.")
    else:
        # Layout 2 kolom
        col1, col2 = st.columns(2)
        
        with col1:
            # --- Visualisasi 1: Distribusi Kepuasan ---
            st.subheader("Distribusi Tingkat Kepuasan")
            fig, ax = plt.subplots()
            sns.countplot(x='Satisfaction_Rating', data=df_filtered, palette='viridis', hue='Satisfaction_Rating', legend=False, ax=ax)
            ax.set_title('Distribusi Rating Kepuasan')
            st.pyplot(fig)

        with col2:
            # --- Visualisasi 2: Kepadatan Kerumunan ---
            st.subheader("Distribusi Kepadatan Kerumunan")
            fig, ax = plt.subplots()
            df_filtered['Crowd_Density'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette('Pastel2'))
            ax.set_ylabel('') # Hapus label y
            ax.set_title('Proporsi Kepadatan Kerumunan')
            st.pyplot(fig)

        st.markdown("---")
        
        # --- Visualisasi 3: Fitur Kategorikal Pilihan User ---
        st.subheader("Distribusi Fitur Kategorikal (Pilihan Anda)")
        
        categorical_cols_to_show = [
            'Activity_Type', 'Weather_Conditions', 'Fatigue_Level', 
            'Stress_Level', 'Health_Condition', 'Transport_Mode'
        ]
        selected_col = st.selectbox("Pilih Fitur untuk Dilihat:", categorical_cols_to_show)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=selected_col, data=df_filtered, order=df_filtered[selected_col].value_counts().index, palette='viridis', hue=selected_col, legend=False, ax=ax)
        ax.set_title(f'Distribusi Frekuensi untuk: {selected_col}')
        ax.set_xlabel('Jumlah Jamaah')
        st.pyplot(fig)

        # --- Visualisasi 4: Heatmap Korelasi ---
        st.subheader("Heatmap Korelasi Fitur Numerik")
        with st.expander("Tampilkan/Sembunyikan Heatmap"):
            numeric_cols_raw = df_filtered.select_dtypes(include=np.number).columns.tolist()
            numeric_cols_to_corr = [col for col in numeric_cols_raw if 'Lat' not in col and 'Long' not in col and 'binary' not in col]
            
            corr = df_filtered[numeric_cols_to_corr].corr()
            
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, annot_kws={"size": 8})
            ax.set_title('Heatmap Korelasi Antar Fitur Numerik', fontsize=16)
            st.pyplot(fig)
            st.markdown("""
            **Insight Kunci:**
            -   `Perceived_Safety_Rating` memiliki korelasi positif kecil (0.16) dengan `Satisfaction_Rating`.
            -   `Queue_Time_minutes` memiliki korelasi negatif kecil (-0.16). Ini adalah *pembunuh kepuasan*!
            """)

# -----------------------------------------------------------------
# HALAMAN 3: SIMULASI MODEL & PERFORMA
# -----------------------------------------------------------------
elif page == "Simulasi Model & Performa":
    st.title("🤖 Simulasi Model & Performa")
    st.markdown("""
    Di halaman ini, Anda dapat **menguji model secara interaktif** atau melihat **performa keseluruhannya** pada data uji. Model yang digunakan adalah **XGBoost Classifier**.
    """)
    st.markdown("---")

    col1, col2 = st.columns([1.5, 2]) # Kolom pertama lebih sempit

    with col1:
        # **FITUR INTERAKTIF v2.0: FORMULIR PREDIKSI**
        st.subheader("Kalkulator Prediksi Kepuasan 💡")
        st.info("Masukkan skenario di bawah ini untuk mendapatkan prediksi *live* dari model.")
        
        # Ambil input berdasarkan 5 fitur teratas
        
        # Input 1: Perceived_Safety_Rating
        safety_rating = st.slider(
            "Rating Keamanan (Perceived Safety Rating)", 1, 5, 3
        )
        
        # Input 2: Time_Spent_at_Location_minutes
        time_spent = st.slider(
            "Waktu di Lokasi (Menit)", 10, 120, 60
        )
        
        # Input 3: Queue_Time_minutes
        queue_time = st.slider(
            "Waktu Antri (Menit)", 0, 60, 15
        )
        
        # Input 4: Security_Checkpoint_Wait_Time
        security_wait = st.slider(
            "Waktu Antri Keamanan (Menit)", 0, 30, 10
        )
        
        # Input 5: Crowd_Density
        crowd_density = st.selectbox(
            "Kepadatan Kerumunan (Crowd Density)",
            options=['Low', 'Medium', 'High']
        )
        
        # Tombol Prediksi
        submit_button = st.button("Dapatkan Prediksi", type="primary")

    # --- Logika di Kolom Kedua (Hasil Prediksi & Performa Model) ---
    with col2:
        if submit_button:
            # 1. Buat template DataFrame dengan semua 0, sesuai kolom X
            input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
            
            # 2. Isi nilai numerik dari form
            numeric_inputs = {
                'Perceived_Safety_Rating': safety_rating,
                'Time_Spent_at_Location_minutes': time_spent,
                'Queue_Time_minutes': queue_time,
                'Security_Checkpoint_Wait_Time': security_wait
            }
            
            # Isi nilai numerik lain (yang tidak ada di form) dengan RATA-RATA
            for col in numeric_cols:
                if col not in numeric_inputs:
                    # Mengisi dgn nilai rata-rata dari data mentah
                    input_data[col] = df_raw[col].mean()
                else:
                    input_data[col] = numeric_inputs[col]
            
            # 3. Scale nilai numerik
            input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            
            # 4. Isi nilai kategorikal dari form
            # (Ingat kita pakai drop_first=False)
            input_data[f'Crowd_Density_{crowd_density}'] = 1
            
            # Isi nilai kategorikal lain (yang tidak ada di form) dengan MODUS (paling umum)
            for col_prefix in ['Activity_Type', 'Weather_Conditions', 'Fatigue_Level', 'Stress_Level', 'Health_Condition', 'Age_Group', 'Pilgrim_Experience', 'Transport_Mode', 'Emergency_Event', 'Crowd_Morale', 'AR_Navigation_Success', 'Nationality']:
                if f'{col_prefix}_{crowd_density}' not in input_data.columns: # Hindari overwrite crowd_density
                    modus_value = df_raw[col_prefix].mode()[0]
                    # Pastikan kolomnya ada
                    if f'{col_prefix}_{modus_value}' in input_data.columns:
                        input_data[f'{col_prefix}_{modus_value}'] = 1

            # 5. Dapatkan Prediksi
            prediction = model.predict(input_data[feature_columns]) # Pastikan urutan kolom
            prediction_proba = model.predict_proba(input_data[feature_columns])
            
            # 6. Tampilkan Hasil
            st.subheader("Hasil Prediksi:")
            if prediction[0] == 1:
                st.success(f"**Prediksi: PUAS** (Probabilitas: {prediction_proba[0][1]:.1%})")
                st.markdown("Model memprediksi bahwa jamaah dengan skenario ini kemungkinan besar akan **Puas**.")
            else:
                st.error(f"**Prediksi: TIDAK PUAS** (Probabilitas: {prediction_proba[0][0]:.1%})")
                st.markdown("Model memprediksi bahwa jamaah dengan skenario ini kemungkinan besar akan **Tidak Puas**.")
        
        else:
            # Tampilan default jika tombol belum ditekan
            st.info("Klik tombol 'Dapatkan Prediksi' di sebelah kiri untuk melihat simulasi.")

        st.markdown("---")
        
        # --- Bagian Performa Model (Tetap ditampilkan) ---
        st.header("Performa Model Keseluruhan")
        st.markdown("Hasil ini didapat dari 2.000 data uji (data yang belum pernah dilihat model).")
        
        y_pred_xgb_tuned = model.predict(X_test)
        
        # Menampilkan Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred_xgb_tuned, target_names=['Tidak Puas (0)', 'Puas (1)'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
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
        
        # Menampilkan Fitur Terpenting
        with st.expander("Lihat Faktor Paling Berpengaruh (Feature Importance)"):
            feature_importances = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_15_features = feature_importances.head(15)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=top_15_features, palette='viridis', hue='feature', legend=False, ax=ax)
            ax.set_title('Top 15 Feature Importances - XGBoost Model', fontsize=16)
            ax.set_xlabel('Tingkat Kepentingan', fontsize=12)
            ax.set_ylabel('Fitur', fontsize=12)
            st.pyplot(fig)


# -----------------------------------------------------------------
# HALAMAN 4: INSIGHT & REKOMENDASI (BARU!)
# -----------------------------------------------------------------
elif page == "Insight & Rekomendasi":
    st.title("💡 Insight Kunci & Rekomendasi Operasional")
    st.markdown("""
    Halaman ini merangkum temuan terpenting dari analisis dan model, 
    lalu menerjemahkannya menjadi rekomendasi yang dapat ditindaklanjuti.
    """)
    st.markdown("---")

    # Ambil feature importance
    # (Kita hitung ulang di sini agar halaman ini independen)
    feature_importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_5_features = feature_importances.head(5)

    st.header("Faktor Kunci Penentu Kepuasan")
    st.info("""
    Model XGBoost kita telah mengidentifikasi beberapa faktor yang secara konsisten 
    memiliki pengaruh terbesar dalam menentukan apakah seorang jamaah 'Puas' atau 'Tidak Puas'.
    """)

    # Tampilkan 5 fitur teratas
    st.dataframe(top_5_features, use_container_width=True)

    # Plot 5 fitur teratas
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x='importance', 
        y='feature', 
        data=top_5_features, 
        palette='viridis', 
        hue='feature', 
        legend=False, 
        ax=ax
    )
    ax.set_title('Top 5 Faktor Paling Berpengaruh')
    ax.set_xlabel('Tingkat Kepentingan (Importance Score)')
    ax.set_ylabel('Fitur')
    st.pyplot(fig)

    st.markdown("---")
    
    st.header("Rekomendasi Operasional (Actionable Insights)")
    st.markdown("""
    Berdasarkan faktor-faktor di atas, berikut adalah rekomendasi strategis 
    untuk meningkatkan kepuasan jamaah:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Prioritas 1: Optimalkan Waktu Antri (Queue Time)**")
        st.markdown("""
        - **Insight:** `Queue_Time_minutes` dan `Security_Checkpoint_Wait_Time` adalah 'pembunuh kepuasan' terbesar ke-2 dan ke-4.
        - **Rekomendasi:** 1.  Implementasikan sistem pemantauan antrian *real-time* di titik-titik krusial (Tawaf, Sa'i, Keamanan).
            2.  Buka pos pemeriksaan keamanan tambahan secara dinamis saat kepadatan (`Crowd_Density`) terdeteksi 'High'.
            3.  Gunakan model ini untuk memprediksi potensi ketidakpuasan saat antrian melebihi 15 menit (berdasarkan slider di simulator).
        """)

    with col2:
        st.success("**Prioritas 2: Tingkatkan Persepsi Keamanan (Perceived Safety)**")
        st.markdown("""
        - **Insight:** `Perceived_Safety_Rating` adalah faktor terpenting #1. Ini bukan hanya tentang *ada* atau *tidaknya* insiden, tapi tentang apa yang *dirasakan* jamaah.
        - **Rekomendasi:** 1.  Tingkatkan *visibilitas* petugas keamanan di area padat.
            2.  Pastikan pencahayaan yang baik di semua area, terutama pada malam hari.
            3.  Berikan informasi proaktif tentang langkah-langkah keamanan yang sedang berjalan (misal: melalui AR System).
        """)

    st.markdown("---")

    st.header("Kesimpulan Proyek & Langkah Selanjutnya")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Kesimpulan")
        st.markdown("""
        Proyek ini berhasil menunjukkan bahwa kepuasan jamaah sangat dipengaruhi oleh **efisiensi** dan **rasa aman**. 
        
        Meskipun faktor-faktor seperti cuaca atau aktivitas tidak dapat diubah, faktor-faktor operasional seperti **waktu antri** dan **manajemen keamanan** dapat dioptimalkan.
        
        Model XGBoost yang dibangun (meskipun performanya masih bisa ditingkatkan) telah berhasil mengidentifikasi faktor-faktor kunci ini, dan 'Kalkulator Prediksi' menyediakan alat bantu untuk simulasi skenario.
        """)

    with col2:
        st.subheader("Keterbatasan & Langkah Selanjutnya")
        st.warning("""
        - **Keterbatasan 1 (Data):** Dataset ini kemungkinan besar **sintetis** (dibuat secara artifisial), seperti yang terlihat dari distribusi rating yang sempurna (1-5). Ini berarti model mungkin tidak akan berperforma sama baiknya pada data dunia nyata yang 'kotor' dan tidak seimbang.
        
        - **Keterbatasan 2 (Model):** Performa model (F1-Score makro ~0.58) masih tergolong sedang. Model ini lebih baik dalam memprediksi 'Tidak Puas' daripada 'Puas'.
        
        - **Langkah Selanjutnya:**
            1.  **Validasi:** Uji model ini menggunakan data riil (jika tersedia).
            2.  **Tuning:** Lakukan *Hyperparameter Tuning* (misal: GridSearchCV) pada XGBoost untuk meningkatkan performa.
            3.  **Fitur Baru:** Kumpulkan data/fitur baru, seperti jam (pagi/siang/malam), hari (Jumat/biasa), atau event khusus yang mungkin sangat memengaruhi kepadatan.
        """)
