import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import xgboost as xgb
    import time
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from string import punctuation
    import io
    import numpy as np

    # --- KONFIGURASI HALAMAN & TEMA (DARI .streamlit/config.toml) ---
    st.set_page_config(
        page_title="Analisis Crowd Management Haji",
        page_icon="🕋",
        layout="wide"
    )
    
    # Tema kustom (cara sederhana tanpa file config.toml)
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
        .stApp .stButton>button {
            background-color: #00A0FF;
            color: #0A0F1F;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- FUNGSI BANTUAN ---
    
    # Download NLTK (di-cache)
    @st.cache_resource
    def download_nltk_data():
        try: nltk.data.find('tokenizers/punkt')
        except LookupError: nltk.download('punkt')
        try: nltk.data.find('corpora/stopwords')
        except LookupError: nltk.download('stopwords')
    download_nltk_data()

    # Load Data (di-cache)
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('hajj_umrah_crowd_management_dataset.csv')
        except FileNotFoundError:
            st.error("Pastikan file 'hajj_umrah_crowd_management_dataset.csv' ada di folder yang sama.")
            return pd.DataFrame(), [], []
        
        target = 'Satisfaction_Rating'
        numericals = df.select_dtypes(include=['int64', 'float64']).columns.drop(target, errors='ignore')
        categoricals = df.select_dtypes(include=['object']).columns.drop(['Timestamp', 'Health_Condition', 'Emergency_Event', 'Incident_Type', 'Crowd_Morale', 'Event_Type'], errors='ignore')
        return df, numericals, categoricals

    # Fungsi Konversi (di-cache)
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    @st.cache_data
    def convert_fig_to_png(fig):
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        return buf.getvalue()

    # Fungsi NLP (di-cache)
    @st.cache_data
    def get_summary(text, top_n=3):
        try:
            stop_words = set(stopwords.words('english') + list(punctuation))
            sentences = sent_tokenize(text)
            if not sentences: return "Teks tidak valid."
            words = word_tokenize(text.lower())
            word_freq = {}
            for word in words:
                if word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            if not word_freq: return "Teks tidak mengandung kata yang signifikan."
            max_freq = max(word_freq.values())
            for word in word_freq.keys(): word_freq[word] = (word_freq[word] / max_freq)
            sentence_scores = {}
            for sent in sentences:
                for word in word_tokenize(sent.lower()):
                    if word in word_freq:
                        if len(sent.split(' ')) < 30:
                            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]
            summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
            return ' '.join(summary_sentences)
        except Exception as e: return f"Error saat meringkas: {e}"

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
        
        # --- BUAT TABS (INI PENGGANTI FOLDER 'PAGES') ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Home", 
            "📊 Data Overview", 
            "📈 EDA", 
            "🤖 Modeling", 
            "💡 Kesimpulan"
        ])
        
        # --- KONTEN TAB 1: HOME ---
        with tab1:
            st.header("Selamat Datang")
            st.markdown("Proyek ini awalnya bertujuan untuk memprediksi `Satisfaction_Rating` jamaah haji.")
            st.markdown("Namun, analisis mendalam mengungkap *insight* yang lebih penting: **pentingnya validasi data**.")
            st.subheader("Narasi Proyek (Plot Twist)")
            st.warning("""
            **HIPOTESIS AWAL:** Kita bisa memprediksi kepuasan jamaah.\n
            **PROSES:** Melakukan EDA dan membangun model (XGBoost).\n
            **HASIL:** Model memiliki performa sangat rendah (F1-Score ~0.39).\n
            **INVESTIGASI 'MENGAPA?':**\n
            1.  Distribusi Target (Kepuasan 1-5) **seimbang sempurna** (~2000 sampel per kelas).\n
            2.  Data **100% bersih** (0 *missing values*, 0 *duplicates*).\n
            3.  Tidak ada *outlier* yang logis.\n
            **KESIMPULAN AKHIR:** Dataset ini adalah **data sintetis** (buatan) dan tidak cocok untuk *deployment* di dunia nyata. Keberhasilan proyek ini adalah **membuktikan secara analitis** bahwa data ini tidak dapat dipercaya.
            """)
            st.markdown("Silakan navigasi ke *tab* lain untuk melihat bukti analisisnya.")

        # --- KONTEN TAB 2: DATA OVERVIEW ---
        with tab2:
            st.header("📊 1. Data Overview & Kualitas Data")
            st.markdown("Dataset ini berisi **30 fitur** dan **10.000 baris** data.")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                label="Unduh Dataset Mentah (.csv)",
                data=convert_df_to_csv(df),
                file_name="hajj_umrah_crowd_dataset.csv",
                mime="text/csv",
            )
            st.markdown("---")
            st.header("Pengecekan Kualitas Data")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Hilang (Missing Values)")
                st.metric("Jumlah Data Hilang", df.isnull().sum().sum())
                st.info("**Insight:** Tidak ada data hilang sama sekali.")
            with col2:
                st.subheader("Data Duplikat")
                st.metric("Jumlah Baris Duplikat", df.duplicated().sum())
                st.info("**Insight:** Tidak ada data duplikat.")

        # --- KONTEN TAB 3: EDA (EXPLORATORY DATA ANALYSIS) ---
        with tab3:
            st.header("📈 2. Exploratory Data Analysis (EDA)")
            st.subheader("Analisis Variabel Target: `Satisfaction_Rating`")
            st.markdown("Ini adalah bukti kunci pertama:")
            fig_target = px.histogram(df, x='Satisfaction_Rating', title='Distribusi Skor Kepuasan (1-5)', color='Satisfaction_Rating', text_auto=True)
            st.plotly_chart(fig_target, use_container_width=True)
            st.warning("**Insight Kunci:** Distribusi yang seimbang sempurna ini (masing-masing 20%) sangat tidak realistis untuk data survei dunia nyata, yang biasanya miring (skewed).")
            
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

        # --- KONTEN TAB 4: PREDICTIVE MODELING ---
        with tab4:
            st.header("🤖 3. Predictive Modeling (XGBoost)")
            st.markdown("Kita tetap mencoba membangun model untuk membuktikan hipotesis bahwa data ini tidak memiliki sinyal prediktif yang kuat.")
            
            if st.button("Latih Model XGBoost (Mungkin butuh 1-2 menit)", type="primary"):
                with st.spinner("Sedang melatih model..."):
                    # Persiapan Data
                    X = df.drop('Satisfaction_Rating', axis=1)[numericals.tolist() + categoricals.tolist()]
                    y = df['Satisfaction_Rating']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # Preprocessing Pipeline
                    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
                    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, numericals),
                            ('cat', categorical_transformer, categoricals)
                        ])
                    
                    # Model Pipeline
                    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
                    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])
                    
                    # Latih & Prediksi
                    model_pipeline.fit(X_train, y_train)
                    y_pred = model_pipeline.predict(X_test)
                    
                    st.success("Model berhasil dilatih.")
                    
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=False)
                    st.code(report)
                    st.error("**Analisis Hasil:** Performa model sangat rendah (F1-Score makro ~0.20-0.30). Ini mengkonfirmasi bahwa fitur-fitur yang ada tidak dapat memprediksi target yang (kemungkinan besar) dibuat secara acak.")
            else:
                st.info("Klik tombol di atas untuk memulai proses pelatihan model.")

        # --- KONTEN TAB 5: KESIMPULAN & NLP ---
        with tab5:
            st.header("💡 4. Kesimpulan Proyek")
            st.error("Proyek ini adalah studi kasus tentang **mengapa *Data Scientist* tidak boleh memercayai data begitu saja**.")
            st.success("Keberhasilan proyek ini bukanlah pada akurasi model, tapi pada **pembuktian analitis** bahwa dataset ini **sintetis** dan tidak boleh digunakan untuk keputusan bisnis.")
            
            st.markdown("---")
            st.header("Bonus: Fitur NLP Ringkasan Teks")
            st.markdown("Sebagai fitur tambahan, berikut adalah alat peringkas teks sederhana.")
            text_to_summarize = st.text_area("Masukkan teks (Bahasa Inggris) untuk diringkas:", "Streamlit is an open-source Python library...", height=200)
            if st.button("Ringkas Teks Ini"):
                with st.spinner("Meringkas..."):
                    summary = get_summary(text_to_summarize)
                    st.subheader("Hasil Ringkasan:")
                    st.success(summary)
    else:
        st.error("Gagal memuat data. Pastikan 'hajj_umrah_crowd_management_dataset.csv' ada di folder yang benar.")