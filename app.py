import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import pickle
import datetime
import os

# ===================================================================================
# Konfigurasi Halaman Streamlit
# ===================================================================================
st.set_page_config(
    page_title="Prediksi Sewa Sepeda COGNIDATA",
    page_icon="🚲",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'mailto:fauzanzahid720@gmail.com', # Format mailto: sudah benar
        'Report a bug': "mailto:fauzanzahid720@gmail.com",
        'About': "### Aplikasi Prediksi Permintaan Sepeda\nTim COGNIDATA\nPowered by XGBoost & Scikit-learn." # Diubah dari PyCaret ke Scikit-learn
    }
)

# ===================================================================================
# Muat Model
# ===================================================================================
@st.cache_resource
def load_pickled_model(model_path):
    """Memuat model dari file pickle."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model berhasil dimuat dari: {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file ada di direktori yang sama dengan aplikasi.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# PERHATIKAN: Nama file model diubah sesuai dengan penyimpanan pickle murni
MODEL_FILENAME = 'XGBoost_SKLearn_Pipeline.pkl'
pipeline_model = load_pickled_model(MODEL_FILENAME)

# ===================================================================================
# HTML Templates
# ===================================================================================
PRIMARY_BG_COLOR = "#003366"
PRIMARY_TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#FFD700"

HTML_BANNER = f"""
    <div style="background-color:{PRIMARY_BG_COLOR};padding:20px;border-radius:10px;margin-bottom:25px;box-shadow: 0 4px 12px 0 rgba(0,0,0,0.3);">
        <h1 style="color:{PRIMARY_TEXT_COLOR};text-align:center;font-family: 'Verdana', sans-serif;">🚲 Aplikasi Prediksi Permintaan Sewa Sepeda</h1>
        <h4 style="color:{ACCENT_COLOR};text-align:center;font-family: 'Verdana', sans-serif;">Dipersembahkan oleh Tim COGNIDATA</h4>
    </div>
    """

HTML_FOOTER = f"""
    <div style="padding:10px;margin-top:40px;text-align:center;">
        <p style="color:grey;font-size:0.9em;">
            &copy; {datetime.date.today().year} Tim COGNIDATA - Prediksi Permintaan Sepeda
        </p>
    </div>
    """
# ===================================================================================
# Fungsi Utama Aplikasi
# ===================================================================================
def main():
    stc.html(HTML_BANNER, height=170)
    
    menu_options = {
        "🏠 Beranda": show_homepage,
        "⚙️ Aplikasi Prediksi": run_prediction_app,
        "📖 Info Model": show_model_info_page
    }
    
    st.sidebar.title("Navigasi Aplikasi")
    choice = st.sidebar.radio("", list(menu_options.keys()), label_visibility="collapsed")

    menu_options[choice]()
    
    stc.html(HTML_FOOTER, height=70)

# ===================================================================================
# Halaman Beranda
# ===================================================================================
def show_homepage():
    st.markdown("## Selamat Datang di Dasbor Prediksi Permintaan Sepeda!")
    st.markdown("""
    Aplikasi ini adalah alat bantu cerdas untuk memprediksi jumlah total sepeda yang kemungkinan akan disewa dalam satu jam tertentu. 
    Dengan memanfaatkan data historis dan model machine learning canggih, kami bertujuan untuk memberikan estimasi yang dapat diandalkan 
    untuk membantu Anda dalam perencanaan dan operasional bisnis berbagi sepeda.

    ---
    #### Mengapa Prediksi Ini Penting?
    - Optimalisasi Stok
    - Efisiensi Operasional
    - Peningkatan Kepuasan Pelanggan
    - Strategi Pemasaran

    ---
    #### Cara Kerja Aplikasi:
    1.  Pilih "**⚙️ Aplikasi Prediksi**" dari menu navigasi di sebelah kiri.
    2.  Masukkan detail parameter pada formulir yang disediakan.
    3.  Klik tombol "**Prediksi Sekarang**" untuk melihat estimasi jumlah sewa.
    
    Jelajahi juga halaman "**📖 Info Model**" untuk memahami lebih dalam tentang teknologi di balik prediksi ini.

    ---
    #### Sumber Data:
    Dataset yang digunakan dalam pengembangan model ini berasal dari kompetisi Kaggle:
    [Bike Sharing Demand - Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data)
    """)
    
    st.image("https://img.freepik.com/free-photo/row-parked-rental-bikes_53876-63261.jpg", 
             caption="Inovasi Transportasi Perkotaan dengan Berbagi Sepeda", use_column_width=True)

# ===================================================================================
# Halaman Aplikasi Prediksi (dengan Tata Letak Kolom yang Diperjelas)
# ===================================================================================
def run_prediction_app():
    st.markdown("## ⚙️ Masukkan Parameter untuk Prediksi")
    
    if pipeline_model is None:
        st.error("Model prediksi tidak dapat dimuat. Silakan periksa file model dan coba lagi.")
        return

    # --- Bagian Input Tanggal dan Waktu ---
    st.markdown("#### 📅 Informasi Waktu")
    col_date, col_time = st.columns([1, 1]) # Rasio kolom 1:1
    with col_date:
        input_date = st.date_input("Tanggal Prediksi", datetime.date.today() + datetime.timedelta(days=1), help="Pilih tanggal untuk prediksi.")
    with col_time:
        input_time = st.time_input("Waktu Prediksi", datetime.time(10, 0), help="Pilih waktu (jam & menit) untuk prediksi.")
    dt_object = datetime.datetime.combine(input_date, input_time)
    
    is_working_day_auto = 1 if dt_object.weekday() < 5 else 0 
    workingday_display_text = "Hari Kerja" if is_working_day_auto == 1 else "Akhir Pekan/Libur"
    st.info(f"Prediksi untuk: **{dt_object.strftime('%A, %d %B %Y, pukul %H:%M')}** ({workingday_display_text})")
    
    st.markdown("---")

    st.markdown("#### 📋 Kondisi & Lingkungan")
    # Menggunakan rasio yang sedikit berbeda untuk 3 kolom fitur
    col_kondisi1, col_kondisi2, col_lingkungan = st.columns([2, 2, 2.5]) 

    with col_kondisi1: 
        st.markdown("##### Musim & Liburan")
        season_options = {1: "Musim Dingin", 2: "Musim Semi", 3: "Musim Panas", 4: "Musim Gugur"}
        current_month = dt_object.month
        if current_month in [12, 1, 2]: default_season = 1
        elif current_month in [3, 4, 5]: default_season = 2
        elif current_month in [6, 7, 8]: default_season = 3
        else: default_season = 4
        season = st.selectbox("Musim", options=list(season_options.keys()), 
                              format_func=lambda x: f"{season_options[x]} ({x})", 
                              index=list(season_options.keys()).index(default_season),
                              key="season_select")
        
        holiday = st.radio("Hari Libur?", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak", 
                           index=0, horizontal=True, key="holiday_radio")

    with col_kondisi2: 
        st.markdown("##### Status Hari & Cuaca")
        workingday = st.radio("Hari Kerja Aktual?", (0, 1), 
                              format_func=lambda x: "Ya" if x == 1 else "Tidak", 
                              index=is_working_day_auto, horizontal=True, key="workingday_radio",
                              help=f"Terdeteksi otomatis sebagai '{workingday_display_text}', Anda bisa mengubahnya jika perlu.")
        
        weather_options = {1: "Cerah/Cerah Berawan", 2: "Kabut/Berawan", 3: "Hujan/Salju Ringan", 4: "Cuaca Ekstrem"}
        weather = st.selectbox("Kondisi Cuaca", options=list(weather_options.keys()), 
                               format_func=lambda x: f"{weather_options[x]} ({x})", 
                               index=0, key="weather_select")

    with col_lingkungan: 
        st.markdown("##### Parameter Lingkungan")
        temp = st.number_input("Suhu (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5, format="%.1f", key="temp_input")
        humidity = st.slider("Kelembapan (%)", min_value=0, max_value=100, value=60, step=1, key="humidity_slider")
        windspeed = st.number_input("Kecepatan Angin (km/jam)", min_value=0.0, max_value=80.0, value=10.0, step=0.1, format="%.1f", key="windspeed_input")

    st.markdown("---")
    
    if st.button("Prediksi Jumlah Sewa Sekarang!", use_container_width=True, type="primary", key="predict_button_main"):
        if pipeline_model is not None:
            input_features = pd.DataFrame([{
                'datetime': dt_object, 'season': season, 'holiday': holiday,
                'workingday': workingday, 'weather': weather, 'temp': temp,
                'atemp': temp, 'humidity': humidity, 'windspeed': windspeed
            }])
            
            st.markdown("#### Hasil Prediksi")
            try:
                prediction_value = pipeline_model.predict(input_features)
                predicted_count_final = max(0, int(round(prediction_value[0])))
                
                st.metric(label="Estimasi Jumlah Sewa Sepeda", value=f"{predicted_count_final} unit")

                if predicted_count_final < 50:
                    st.info("Saran: Permintaan diprediksi rendah.")
                elif predicted_count_final < 250:
                    st.success("Saran: Permintaan diprediksi sedang.")
                else:
                    st.warning("Saran: Permintaan diprediksi tinggi.")
            except Exception as e:
                st.error(f"Gagal membuat prediksi: {e}")
        else:
            st.error("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")
            
#====================================================================================#
# Halaman Informasi Model
#====================================================================================#
def show_model_info_page():
    st.markdown("## 📖 Informasi Detail Model Prediksi")
    st.markdown(f"""
    Model prediktif yang menjadi tulang punggung aplikasi ini adalah **XGBoost Regressor** yang dipaketkan dalam pipeline Scikit-learn.
    Model ini awalnya dikembangkan dan dioptimalkan menggunakan _framework_ PyCaret, kemudian pipeline finalnya disimpan untuk penggunaan mandiri.

    #### Arsitektur & Pra-pemrosesan:
    Model yang Anda gunakan (`{MODEL_FILENAME}`) adalah **keseluruhan pipeline pra-pemrosesan Scikit-learn dan model XGBoost** yang telah di-*fit* pada data historis. Proses ini mencakup berbagai langkah seperti:
    - Ekstraksi fitur waktu dari kolom `datetime`.
    - Rekayasa fitur siklikal (sin/cos) untuk menangkap pola periodik.
    - Penanganan pencilan (winsorizing) untuk fitur seperti `humidity` dan `windspeed`.
    - Scaling fitur numerik.
    - Encoding fitur kategorikal (One-Hot Encoding).
    - Transformasi target (misalnya, Yeo-Johnson) untuk menormalkan distribusi.

    #### Sumber Data Acuan:
    Model ini dikembangkan berdasarkan konsep dan data dari kompetisi Kaggle:
    [Bike Sharing Demand - Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data)

    #### Performa Model (Contoh dari Sesi Pelatihan Awal dengan PyCaret):
    - **MAPE (Mean Absolute Percentage Error) pada Hold-out Set**: Sekitar **22.54%**
    - **R² (R-squared) pada Hold-out Set**: Sekitar **0.9614**
    
    *Metrik ini menunjukkan kemampuan prediksi model, namun performa pada data baru dapat bervariasi.*
    """)
    
    if pipeline_model is not None:
        st.markdown("#### Detail Pipeline dan Parameter Estimator Inti (XGBoost):")
        st.write("Struktur Pipeline Model:")
        st.write(pipeline_model) # Menampilkan struktur pipeline

        try:
            # Mengakses model XGBoost sebenarnya dari dalam pipeline scikit-learn
            # Ini mungkin perlu disesuaikan jika pipeline Anda memiliki struktur yang berbeda
            # atau jika model dibungkus oleh TransformedTargetRegressor
            actual_model_estimator = None
            if hasattr(pipeline_model, 'steps'): # Jika ini objek Pipeline
                final_step_estimator = pipeline_model.steps[-1][1]
                if hasattr(final_step_estimator, 'regressor'): # Jika dibungkus TransformedTargetRegressor
                    actual_model_estimator = final_step_estimator.regressor
                else: # Jika langkah terakhir adalah model itu sendiri
                    actual_model_estimator = final_step_estimator
            elif hasattr(pipeline_model, 'regressor'): # Jika pipeline_model adalah TransformedTargetRegressor itu sendiri
                 actual_model_estimator = pipeline_model.regressor
            else: # Jika pipeline_model adalah estimator itu sendiri
                actual_model_estimator = pipeline_model

            if actual_model_estimator and hasattr(actual_model_estimator, 'get_params'):
                st.markdown("Parameter Model XGBoost:")
                st.json(actual_model_estimator.get_params(), expanded=False)
            else:
                st.warning("Tidak dapat mengekstrak parameter model XGBoost dari pipeline.")
        except Exception as e:
            st.warning(f"Tidak dapat menampilkan parameter model secara detail: {e}")
    
    st.info("Untuk detail teknis lebih lanjut, silakan merujuk pada dokumentasi pengembangan internal Tim COGNIDATA.")

#====================================================================================#
# Menjalankan Aplikasi
#====================================================================================#
if __name__ == "__main__":
    if pipeline_model is None:
        st.error("KRITIS: GAGAL MEMUAT MODEL PREDIKSI.")
        st.markdown(f"Pastikan file model `{MODEL_FILENAME}` ada di direktori yang sama dengan `app.py`.")
        st.stop()
    main()