import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # [OPTIMASI] Import Plotly Express
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
# PENGATURAN HALAMAN
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Kepuasan Jamaah",
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------
# FUNGSI-FUNGSI (Tidak berubah, kita sembunyikan dgn st.expander)
# -----------------------------------------------------------------
with st.expander("Lihat Logika Pemuatan Data & Pelatihan Model (Backend)"):
    @st.cache_data
    def load_data():
        df = pd.read_csv("hajj_umrah_crowd_management_dataset.csv")
        df['satisfaction_binary'] = df['Satisfaction_Rating'].apply(lambda x: 1 if x >= 4 else 0)
        
        categorical_cols = [
            'Crowd_Density', 'Activity_Type', 'Weather_Conditions', 'Fatigue_Level',
            'Stress_Level', 'Health_Condition', 'Age_Group', 'Pilgrim_Experience',
            'Transport_Mode', 'Emergency_Event', 'Crowd_Morale', 'AR_Navigation_Success',
            'Nationality'
        ]
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        
        numeric_cols = [
            'Movement_Speed', 'Temperature', 'Sound_Level_dB', 'Queue_Time_minutes',
            'Waiting_Time_for_Transport', 'Security_Checkpoint_Wait_Time',
            'Interaction_Frequency', 'Distance_Between_People_m', 'Time_Spent_at_Location_minutes',
            'Perceived_Safety_Rating'
        ]
        
        scaler = StandardScaler()
        scaler.fit(df_encoded[numeric_cols]) 
        df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
        
        encoded_feature_names = [col for col in df_encoded.columns if any(cat_col in col for cat_col in categorical_cols)]
        feature_columns = numeric_cols + encoded_feature_names
        
        X = df_encoded[feature_columns]
        y = df_encoded['satisfaction_binary']
        
        return df, X, y, scaler, numeric_cols, feature_columns

    df_raw, X, y, scaler, numeric_cols, feature_columns = load_data()

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

    model, X_test, y_test = train_model(X, y)

# -----------------------------------------------------------------
# SIDEBAR (NAVIGASI & FILTER)
# -----------------------------------------------------------------
st.sidebar.title("Navigasi 🧭")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ("Ringkasan Proyek", "Eksplorasi Data (EDA)", "Simulasi Model & Performa", "Insight & Rekomendasi")
)
st.sidebar.markdown("---")

st.sidebar.header("Filter Data (untuk EDA)")
st.sidebar.info("Filter ini akan memengaruhi **semua plot** di halaman 'Eksplorasi Data (EDA)'.")

age_options = st.sidebar.multiselect(
    "Pilih Kelompok Usia:",
    options=df_raw['Age_Group'].unique(),
    default=df_raw['Age_Group'].unique()
)
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

# -----------------------------------------------------------------
# HALAMAN 2: EKSPLORASI DATA (EDA)
# -----------------------------------------------------------------
elif page == "Eksplorasi Data (EDA)":
    st.title("🔍 Eksplorasi Data (EDA) Interaktif")
    
    if df_filtered.empty:
        st.error("Tidak ada data yang cocok dengan filter Anda. Silakan sesuaikan filter di sidebar.")
    else:
        st.info(f"Anda sedang melihat **{len(df_filtered)}** dari **{len(df_raw)}** total data (berdasarkan filter di sidebar).")

        # [OPTIMASI] Menggunakan Tabs untuk desain yang lebih bersih
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distribusi Rating & Kepadatan", 
            "📈 Analisis Mendalam (EDA Baru!)",
            "🌐 EDA Interaktif (Plotly)", 
            "🗺️ Heatmap Korelasi"
        ])
        
        with tab1:
            st.subheader("Distribusi Rating Kepuasan & Kepadatan")
            col1, col2 = st.columns(2)
            
            with col1:
                # [OPTIMASI] Menggunakan container + border untuk visual grouping
                with st.container(border=True):
                    st.markdown("##### Distribusi Tingkat Kepuasan")
                    fig, ax = plt.subplots()
                    sns.countplot(x='Satisfaction_Rating', data=df_filtered, palette='viridis', hue='Satisfaction_Rating', legend=False, ax=ax)
                    # [OPTIMASI] Plot konsisten
                    st.pyplot(fig, use_container_width=True) 
            
            with col2:
                with st.container(border=True):
                    st.markdown("##### Proporsi Kepadatan Kerumunan")
                    fig, ax = plt.subplots()
                    df_filtered['Crowd_Density'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette('Pastel2'))
                    ax.set_ylabel('')
                    st.pyplot(fig, use_container_width=True)

        with tab2:
            # [OPTIMASI] Menjawab "EDA Kurang Banyak"
            st.subheader("Analisis Mendalam: Hubungan Antar Fitur")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.container(border=True):
                    st.markdown("##### Waktu Antri vs Kepadatan")
                    fig, ax = plt.subplots()
                    sns.boxplot(
                        x='Crowd_Density', 
                        y='Queue_Time_minutes', 
                        data=df_filtered, 
                        palette='coolwarm',
                        hue='Crowd_Density',
                        legend=False,
                        order=['Low', 'Medium', 'High'],
                        ax=ax
                    )
                    ax.set_title('Waktu Antri Berdasarkan Kepadatan')
                    ax.set_xlabel('Kepadatan Kerumunan')
                    ax.set_ylabel('Waktu Antri (Menit)')
                    st.pyplot(fig, use_container_width=True)
                    st.markdown("**Insight:** Semakin tinggi kepadatan, waktu antri rata-rata (median) meningkat, dan variasinya (panjang box) juga semakin besar.")
            
            with col2:
                with st.container(border=True):
                    st.markdown("##### Kepuasan vs Tipe Aktivitas")
                    fig, ax = plt.subplots()
                    # Menghitung persentase
                    activity_satisfaction = df_filtered.groupby('Activity_Type')['satisfaction_binary'].mean().reset_index().sort_values(by='satisfaction_binary', ascending=False)
                    
                    sns.barplot(
                        x='satisfaction_binary', 
                        y='Activity_Type', 
                        data=activity_satisfaction, 
                        palette='viridis',
                        hue='Activity_Type',
                        legend=False,
                        ax=ax
                    )
                    ax.set_title('Persentase Kepuasan (Rating 4-5) per Aktivitas')
                    ax.set_xlabel('Proporsi Puas (Rating 4-5)')
                    ax.set_ylabel('Tipe Aktivitas')
                    st.pyplot(fig, use_container_width=True)
                    st.markdown("**Insight:** Aktivitas 'Resting' (Istirahat) memiliki tingkat kepuasan paling rendah, mungkin karena kelelahan atau fasilitas yang kurang memadai.")

        with tab3:
            # [OPTIMASI] Menampilkan plot Plotly yang interaktif
            st.subheader("Analisis Interaktif (Plotly)")
            st.markdown("Arahkan kursor ke plot di bawah ini untuk melihat detail.")
            
            with st.container(border=True):
                st.markdown("##### Waktu Antri vs Waktu di Lokasi (Jitter Plot)")
                # Mengambil sampel agar tidak terlalu berat
                df_sample = df_filtered.sample(min(2000, len(df_filtered)))
                
                fig_plotly = px.scatter(
                    df_sample,
                    x="Time_Spent_at_Location_minutes",
                    y="Queue_Time_minutes",
                    color="Satisfaction_Rating",
                    facet_col="Crowd_Density", # [OPTIMASI] Fitur canggih: 1 plot per kepadatan
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Waktu Antri vs Waktu di Lokasi (berdasarkan Kepadatan)",
                    hover_data=['Activity_Type', 'Age_Group'] # Data tambahan saat hover
                )
                # [OPTIMASI] Memanggil plot plotly
                st.plotly_chart(fig_plotly, use_container_width=True)
                st.markdown("""
                **Insight (dari Plotly):**
                -   Kita bisa melihat bahwa titik `Satisfaction_Rating` rendah (1-2) cenderung berkumpul di area `Queue_Time_minutes` yang tinggi.
                -   Dengan memisahkan berdasarkan kepadatan, kita melihat bahwa di `High` density, titik-titik lebih terkumpul di waktu antri yang lebih tinggi.
                """)

        with tab4:
            st.subheader("Heatmap Korelasi Fitur Numerik")
            with st.container(border=True):
                numeric_cols_raw = df_filtered.select_dtypes(include=np.number).columns.tolist()
                numeric_cols_to_corr = [col for col in numeric_cols_raw if 'Lat' not in col and 'Long' not in col and 'binary' not in col]
                
                corr = df_filtered[numeric_cols_to_corr].corr()
                
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, annot_kws={"size": 8})
                st.pyplot(fig, use_container_width=True)


# -----------------------------------------------------------------
# HALAMAN 3: SIMULASI MODEL & PERFORMA
# -----------------------------------------------------------------
elif page == "Simulasi Model & Performa":
    st.title("🤖 Simulasi Model & Performa")

    # [OPTIMASI] Menggunakan Tabs untuk desain yang lebih bersih
    tab1, tab2 = st.tabs(["🚀 Kalkulator Prediksi (Simulasi)", "📊 Performa Model Mendalam"])
    
    with tab1:
        st.header("Kalkulator Prediksi Kepuasan 💡")
        st.markdown("Masukkan skenario di bawah ini untuk mendapatkan prediksi *live* dari model.")
        
        col1, col2 = st.columns([1.5, 2])
        
        with col1:
            with st.form(key="prediction_form"):
                safety_rating = st.slider("Rating Keamanan (Perceived Safety Rating)", 1, 5, 3)
                time_spent = st.slider("Waktu di Lokasi (Menit)", 10, 120, 60)
                queue_time = st.slider("Waktu Antri (Menit)", 0, 60, 15)
                security_wait = st.slider("Waktu Antri Keamanan (Menit)", 0, 30, 10)
                crowd_density = st.selectbox("Kepadatan Kerumunan (Crowd Density)", options=['Low', 'Medium', 'High'])
                
                # Tombol submit ada di dalam form
                submit_button = st.form_submit_button(label="Dapatkan Prediksi", type="primary")

        with col2:
            if submit_button:
                # Logika prediksi (sama seperti sebelumnya)
                input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
                numeric_inputs = {
                    'Perceived_Safety_Rating': safety_rating,
                    'Time_Spent_at_Location_minutes': time_spent,
                    'Queue_Time_minutes': queue_time,
                    'Security_Checkpoint_Wait_Time': security_wait
                }
                for col in numeric_cols:
                    if col not in numeric_inputs:
                        input_data[col] = df_raw[col].mean()
                    else:
                        input_data[col] = numeric_inputs[col]
                input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
                input_data[f'Crowd_Density_{crowd_density}'] = 1
                for col_prefix in ['Activity_Type', 'Weather_Conditions', 'Fatigue_Level', 'Stress_Level', 'Health_Condition', 'Age_Group', 'Pilgrim_Experience', 'Transport_Mode', 'Emergency_Event', 'Crowd_Morale', 'AR_Navigation_Success', 'Nationality']:
                    if f'{col_prefix}_{crowd_density}' not in input_data.columns:
                        modus_value = df_raw[col_prefix].mode()[0]
                        if f'{col_prefix}_{modus_value}' in input_data.columns:
                            input_data[f'{col_prefix}_{modus_value}'] = 1

                prediction = model.predict(input_data[feature_columns])
                prediction_proba = model.predict_proba(input_data[feature_columns])
                
                # Tampilkan Hasil
                with st.container(border=True):
                    st.subheader("Hasil Prediksi:")
                    if prediction[0] == 1:
                        st.success(f"**Prediksi: PUAS** (Probabilitas: {prediction_proba[0][1]:.1%})")
                    else:
                        st.error(f"**Prediksi: TIDAK PUAS** (Probabilitas: {prediction_proba[0][0]:.1%})")
                    st.markdown("Hasil ini adalah prediksi dari model XGBoost berdasarkan skenario yang Anda masukkan.")
            else:
                st.info("Masukkan skenario di sebelah kiri dan klik 'Dapatkan Prediksi'.")

    with tab2:
        st.header("Performa Model Keseluruhan")
        st.markdown("Hasil ini didapat dari 2.000 data uji (data yang belum pernah dilihat model).")
        
        y_pred_xgb_tuned = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred_xgb_tuned, target_names=['Tidak Puas (0)', 'Puas (1)'], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
        
        with col2:
            with st.container(border=True):
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(
                    model, X_test, y_test, 
                    cmap='Blues', 
                    display_labels=['Tidak Puas', 'Puas'],
                    ax=ax
                )
                st.pyplot(fig, use_container_width=True)
        
        with st.container(border=True):
            st.subheader("Faktor Paling Berpengaruh (Feature Importance)")
            feature_importances = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_15_features = feature_importances.head(15)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=top_15_features, palette='viridis', hue='feature', legend=False, ax=ax)
            st.pyplot(fig, use_container_width=True)


# -----------------------------------------------------------------
# HALAMAN 4: INSIGHT & REKOMENDASI
# -----------------------------------------------------------------
elif page == "Insight & Rekomendasi":
    st.title("💡 Insight Kunci & Rekomendasi Operasional")
    st.markdown("""
    Halaman ini merangkum temuan terpenting dari analisis dan model, 
    lalu menerjemahkannya menjadi rekomendasi yang dapat ditindaklanjuti.
    """)
    st.markdown("---")

    st.header("Faktor Kunci Penentu Kepuasan")
    
    with st.container(border=True):
        # [OPTIMASI] Plot ditampilkan berdampingan dengan teks
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            Model XGBoost kita telah mengidentifikasi beberapa faktor yang secara konsisten 
            memiliki pengaruh terbesar dalam menentukan apakah seorang jamaah 'Puas' atau 'Tidak Puas'.
            """)
            feature_importances = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            top_5_features = feature_importances.head(5)
            st.dataframe(top_5_features, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots()
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
            st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    
    st.header("Rekomendasi Operasional (Actionable Insights)")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.success("**Prioritas 1: Optimalkan Waktu Antri (Queue Time)**")
            st.markdown("""
            - **Insight:** `Queue_Time_minutes` dan `Security_Checkpoint_Wait_Time` adalah 'pembunuh kepuasan' terbesar ke-2 dan ke-4.
            - **Rekomendasi:** 1. Implementasikan sistem pemantauan antrian *real-time*.
                2. Buka pos pemeriksaan keamanan tambahan secara dinamis saat kepadatan 'High'.
                3. Gunakan simulator di dasbor ini untuk menentukan batas waktu antri maksimal sebelum kepuasan anjlok.
            """)

    with col2:
        with st.container(border=True):
            st.success("**Prioritas 2: Tingkatkan Persepsi Keamanan (Perceived Safety)**")
            st.markdown("""
            - **Insight:** `Perceived_Safety_Rating` adalah faktor terpenting #1. Ini bukan hanya tentang *ada* atau *tidaknya* insiden, tapi tentang apa yang *dirasakan* jamaah.
            - **Rekomendasi:** 1. Tingkatkan *visibilitas* petugas keamanan di area padat.
                2. Pastikan pencahayaan yang baik di semua area.
                3. Berikan informasi proaktif tentang langkah-langkah keamanan.
            """)

    st.markdown("---")
    st.header("Kesimpulan Proyek & Langkah Selanjutnya")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Kesimpulan")
            st.markdown("""
            Proyek ini berhasil menunjukkan bahwa kepuasan jamaah sangat dipengaruhi oleh **efisiensi** dan **rasa aman**. 
            Model XGBoost yang dibangun telah berhasil mengidentifikasi faktor-faktor kunci ini, dan 'Kalkulator Prediksi' menyediakan alat bantu untuk simulasi skenario.
            """)
    with col2:
        with st.container(border=True):
            st.subheader("Keterbatasan & Langkah Selanjutnya")
            st.warning("""
            - **Keterbatasan Data:** Dataset ini kemungkinan besar **sintetis**, seperti yang terlihat dari distribusi rating yang sempurna.
            - **Langkah Selanjutnya:**
                1. Validasi model dengan data riil.
                2. Lakukan *Hyperparameter Tuning* (misal: GridSearchCV) untuk meningkatkan performa.
                3. Kembangkan fitur *real-time* berbasis *time-series* (misal: jam padat).
            """)
