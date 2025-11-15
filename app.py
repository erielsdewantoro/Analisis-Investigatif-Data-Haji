import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import numpy as np

# --- KONFIGURASI HALAMAN & TEMA ---
st.set_page_config(
    page_title="Analisis Crowd Management Haji",
    page_icon="🕋",
    layout="wide"
)

# Tema kustom (V1.2 - dengan style untuk Tabs)
st.markdown("""
    <style>
    .stApp {
        background-color: #0A0F1F;
        color: #FAFAFA;
    }
    .stApp [data-testid="stHeader"] {
        font-size: 3rem; font-weight: 700; padding-top: 1rem;
        color: #FAFAFA;
    }
    .stApp h2 { font-weight: 600; color: #FAFAFA; }
    .stApp [data-testid="stSidebar"] {
        background-color: #1E212A;
    }
    .stApp [data-testid="stMetricValue"] { color: #00A0FF; }
    
    /* Style untuk Tabs */
    button[data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        background-color: transparent;
        color: #AAAAAA; /* Warna tab tidak aktif */
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1E212A; /* Warna latar tab aktif */
        color: #00A0FF; /* Warna teks tab aktif */
        border-bottom: 2px solid #00A0FF;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI BANTUAN ---

# Load Data (di-cache)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hajj_umrah_crowd_management_dataset.csv')
    except FileNotFoundError:
        st.error("Pastikan file 'hajj_umrah_crowd_management_dataset.csv' ada di folder yang sama.")
        return pd.DataFrame(), [], []
    
    target = 'Satisfaction_Rating'
    # Perbaikan: Pastikan kolom 'target' tidak error jika tidak ada
    numericals_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if target in numericals_cols:
        numericals = numericals_cols.drop(target, errors='ignore')
    else:
        numericals = numericals_cols

    categoricals_cols = df.select_dtypes(include=['object']).columns
    exclude_cols = ['Timestamp', 'Health_Condition', 'Emergency_Event', 'Incident_Type', 'Crowd_Morale', 'Event_Type']
    categoricals = categoricals_cols.drop(exclude_cols, errors='ignore')
    
    return df, numericals, categoricals

# Fungsi Konversi (di-cache)
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- MULAI APLIKASI ---

df, numericals, categoricals = load_data()

if not df.empty:
    # --- SIDEBAR ---
    st.sidebar.title("Navigasi Proyek DS")
    st.sidebar.image("https://placehold.co/400x200/0A0F1F/00A0FF?text=Proyek+DS+Haji&font=lato", use_column_width=True)
    st.sidebar.info("Aplikasi ini memandu Anda melalui investigasi dataset Hajj Crowd Management.")

    # --- JUDUL UTAMA ---
    st.title("🕋 Studi Kasus: Investigasi Dataset Haji & Umrah")
    st.markdown("Sebuah *deep dive* pada *Data Science*, dari EDA, *Modeling*, hingga *Plot Twist* Data Sintetis.")
    
    # --- BUAT TABS (PERBAIKAN NAMA TAB) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home (Plot Twist)", 
        "📊 Data Overview (Bukti #1)", 
        "📈 EDA (Bukti #2)", 
        "🤖 Investigasi Model (Bukti #3)", 
        "💡 Kesimpulan Final"
    ])
    
    # --- KONTEN TAB 1: HOME ---
    with tab1:
        st.header("Selamat Datang di Investigasi Saya")
        st.markdown("Proyek ini awalnya bertujuan untuk memprediksi `Satisfaction_Rating` jamaah haji.")
        st.markdown("Namun, analisis mendalam mengungkap *insight* yang lebih penting: **pentingnya validasi data**.")
        st.subheader("Narasi Proyek (Plot Twist)")
        st.warning("""
        **HIPOTESIS AWAL:** Kita bisa memprediksi kepuasan jamaah.\n
        **PROSES:** Melakukan EDA dan membangun model (XGBoost).\n
        **HASIL:** Model memiliki performa sangat rendah (F1-Score ~0.39).\n
        **INVESTIGASI 'MENGAPA?':**\n
        1.  **Bukti #1 (Tab Data Overview):** Data **100% bersih** (0 *missing values*, 0 *duplicates*).\n
        2.  **Bukti #2 (Tab EDA):** Distribusi Target (Kepuasan 1-5) **seimbang sempurna**.\n
        3.  **Bukti #3 (Tab Investigasi Model):** Model sekelas XGBoost **gagal menemukan pola**.\n
        **KESIMPULAN AKHIR:** Dataset ini adalah **data sintetis** (buatan) dan tidak cocok untuk *deployment* di dunia nyata. Keberhasilan proyek ini adalah **membuktikan secara analitis** bahwa data ini tidak dapat dipercaya.
        """)
        st.markdown("Silakan navigasi ke *tab* lain untuk melihat bukti analisisnya.")

    # --- KONTEN TAB 2: DATA OVERVIEW (BUKTI #1) ---
    with tab2:
        st.header("📊 1. Data Overview & Kualitas Data (Bukti #1)")
        st.markdown("Dataset ini berisi **30 fitur** dan **10.000 baris** data.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="Unduh Dataset Mentah (.csv)",
            data=convert_df_to_csv(df),
            file_name="hajj_umrah_crowd_dataset.csv",
            mime="text/csv",
        )
        st.markdown("---")
        st.header("Pengecekan Kualitas Data (Bukti #1)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Hilang (Missing Values)")
            st.metric("Jumlah Data Hilang", df.isnull().sum().sum())
            st.info("**Insight:** Tidak ada data hilang sama sekali.")
        with col2:
            st.subheader("Data Duplikat")
            st.metric("Jumlah Baris Duplikat", df.duplicated().sum())
            st.info("**Insight:** Tidak ada data duplikat.")
        st.warning("Data yang 100% sempurna (tanpa *missing values* atau *duplicates*) sangat tidak wajar untuk data survei/sensor dunia nyata.")

    # --- KONTEN TAB 3: EDA (BUKTI #2) ---
    with tab3:
        st.header("📈 2. EDA (Bukti Kunci #2)")
        
        col_eda1, col_eda2 = st.columns([1.2, 1]) 
        
        with col_eda1:
            st.subheader("Analisis Variabel Target: `Satisfaction_Rating`")
            fig_target = px.histogram(df, x='Satisfaction_Rating', title='Distribusi Skor Kepuasan (1-5)', color='Satisfaction_Rating', text_auto=True)
            fig_target.update_layout(showlegend=False)
            st.plotly_chart(fig_target, use_container_width=True)
        
        with col_eda2:
            st.subheader("Mengapa Ini Bukti #2?")
            st.warning("""
            **Ini adalah BUKTI TERKUAT data sintetis.**
            
            Distribusi yang seimbang sempurna ini (masing-masing ~20% per kelas) sangat tidak realistis untuk data survei dunia nyata, yang biasanya miring (skewed).
            
            Ini menunjukkan data ini dibuat secara artifisial agar "seimbang".
            """)
            
        st.markdown("---")
        st.header("Analisis Fitur (Interaktif)")
        st.markdown("Pilih fitur untuk melihat hubungannya dengan Tingkat Kepuasan.")
        col1, col2 = st.columns(2)
        with col1: num_col = st.selectbox("Pilih Fitur Numerik", numericals)
        with col2: cat_col = st.selectbox("Pilih Fitur Kategorikal", categoricals)
        
        col_vis1, col_vis2 = st.columns(2)
        with col_vis1:
            st.subheader(f"`{num_col}` vs. Kepuasan")
            fig_num = px.box(df, x='Satisfaction_Rating', y=num_col, color='Satisfaction_Rating', title=f"Hubungan antara {num_col} dan Kepuasan")
            st.plotly_chart(fig_num, use_container_width=True)
        with col_vis2:
            st.subheader(f"`{cat_col}` vs. Kepuasan")
            fig_cat = px.density_heatmap(df, x=cat_col, y='Satisfaction_Rating', text_auto=True, title=f"Hubungan antara {cat_col} dan Kepuasan")
            st.plotly_chart(fig_cat, use_container_width=True)

    # --- KONTEN TAB 4: INVESTIGASI MODEL (BUKTI #3) ---
    with tab4:
        st.header("🤖 3. Investigasi Model (Bukti #3)")
        
        st.subheader("Hipotesis Investigasi")
        st.info("Jika data ini **nyata**, maka fitur-fitur seperti `Crowd_Density`, `Queue_Time_minutes`, dan `Stress_Level` **seharusnya** memiliki sinyal prediktif yang kuat terhadap `Satisfaction_Rating`.")
        
        st.subheader("Metodologi Eksperimen (Offline)")
        st.markdown("""
        1.  **Notebook:** Saya menggunakan `Final-Project.ipynb` (tersedia di repositori GitHub).
        2.  **Preprocessing:** Saya melakukan *scaling* pada fitur numerik dan *one-hot encoding* pada fitur kategorikal.
        3.  **Model:** Saya melatih model `XGBClassifier` yang sudah di-*tuning* (salah satu model *tree-based* terkuat) untuk memprediksi `Satisfaction_Rating` (yang sudah diubah menjadi 0-4).
        """)
        
        st.subheader("Hasil Eksperimen (Bukti Kegagalan Model)")
        st.markdown("Berikut adalah **hasil (snapshot)** dari `Classification Report` yang didapat dari *notebook*:")
        
        # Ini adalah report statis dari notebook Anda
        st.code("""
              precision    recall  f1-score   support
    
           1       0.25      0.01      0.01       394
           2       0.17      0.00      0.01       408
           3       0.28      0.22      0.24       395
           4       0.24      0.04      0.07       402
           5       0.33      0.82      0.47       401
    
    accuracy                           0.22      2000
   macro avg       0.25      0.22      0.16      2000
weighted avg       0.25      0.22      0.16      2000
        """)
        
        st.error("""
        **Analisis Hasil (Bukti #3):**
        Performa model sangat rendah (F1-Score makro 0.16). 
        
        Ini mengkonfirmasi hipotesis kita: **Fitur-fitur yang ada tidak memiliki korelasi nyata dengan target.**
        
        Model sekuat XGBoost tidak dapat menemukan pola, karena memang **tidak ada pola nyata** untuk ditemukan. Ini memperkuat kesimpulan bahwa data ini sintetis.
        """)

    # --- KONTEN TAB 5: KESIMPULAN FINAL ---
    with tab5:
        st.header("💡 4. Kesimpulan Final Investigasi")
        
        col_k1, col_k2 = st.columns(2)
        
        with col_k1:
            st.error("**Temuan Model:** Model Machine Learning **GAGAL** menemukan pola prediktif yang kuat (F1-Score 0.16).")
        with col_k2:
            st.success("**Temuan Investigasi:** Model gagal bukan karena modelnya, tapi karena **DATASET-NYA SINTETIS**.")
            
        st.markdown("---")
        st.subheader("Rangkuman Tiga Bukti Kunci:")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.warning("**Bukti #1: Kualitas Data Sempurna**")
            st.markdown("(dari Tab Data Overview) 0 *missing values* dan 0 *duplicates*. Terlalu bersih untuk data dunia nyata.")
        with c2:
            st.warning("**Bukti #2: Distribusi Target Sempurna**")
            st.markdown("(dari Tab EDA) Distribusi 5 kelas kepuasan sangat seimbang (~20% per kelas), yang sangat tidak realistis.")
        with c3:
            st.warning("**Bukti #3: Performa Model Gagal**")
            st.markdown("(dari Tab Investigasi Model) Model sekuat XGBoost tidak dapat menemukan sinyal, membuktikan fiturnya tidak berkorelasi nyata dengan target.")
        
        st.markdown("---")
        st.header("Value Proyek Ini bagi Stakeholder")
        st.info("""
        Keberhasilan proyek ini bukanlah pada akurasi model, tapi pada **pembuktian analitis** dan **skeptisisme profesional**.
        
        Ini menunjukkan kemampuan saya untuk:
        1.  Tidak hanya "menjalankan model", tapi **menginvestigasi hasilnya secara kritis**.
        2.  **Melindungi bisnis** dari pengambilan keputusan berdasarkan data yang salah atau sintetis.
        3.  Mengkomunikasikan temuan teknis yang kompleks (kegagalan model) menjadi sebuah *insight* bisnis yang jelas (data tidak valid).
        """)

else:
    st.error("Gagal memuat data. Pastikan 'hajj_umrah_crowd_management_dataset.csv' ada di folder yang benar.")
