
import streamlit as st
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Kelulusan",
    page_icon="🎓"
)

# 2. Custom CSS (Warna UI)
st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background-color: #F4F7FC;
    }

    /* Judul utama */
    h1 {
        color: #17365D !important;
        font-weight: 800;
    }

    /* Subjudul */
    h2, h3 {
        color: #24476B !important;
    }

    /* Teks */
    p, label, span {
        color: #334155;
    }

    /* Input */
    div[data-baseweb="input"] {
        background-color: #FFFFFF;
        border-radius: 8px;
    }

    /* Tombol Prediksi Kelulusan */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #17365D !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9px !important;
        height: 45px !important;
        padding: 0 24px !important;
        font-weight: 600 !important;
        transition: 0.2s ease-in-out;
    }

    /* Memastikan seluruh tulisan tombol berwarna putih */
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button span,
    div[data-testid="stFormSubmitButton"] button div {
        color: #FFFFFF !important;
    }

    /* Efek hover tombol */
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #315F91 !important;
        border: none !important;
        color: #FFFFFF !important;
    }

    /* Efek tulisan saat hover */
    div[data-testid="stFormSubmitButton"] button:hover p,
    div[data-testid="stFormSubmitButton"] button:hover span,
    div[data-testid="stFormSubmitButton"] button:hover div {
        color: #FFFFFF !important;
    }

    /* Efek saat tombol ditekan */
    div[data-testid="stFormSubmitButton"] button:active {
        background-color: #102B4C !important;
    }

    /* Garis pembatas */
    hr {
        border-color: #D5DFEB;
    }

    /* Kotak hasil prediksi */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")

st.write(
    "Aplikasi ini menggunakan model Machine Learning "
    "untuk memprediksi kelulusan."
)


# 3. Load Model
@st.cache_resource
def load_model():
    return joblib.load('model_kelulusan.pkl')


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "File model tidak ditemukan! "
        "Jalankan `1_latih_model.py` terlebih dahulu."
    )
    st.stop()


# 4. Membuat Form Input (UI)
with st.form("form_prediksi"):

    st.subheader("Masukkan Data Mahasiswa")

    ipk = st.number_input(
        "IPK (0.0 - 4.0)",
        min_value=0.0,
        max_value=4.0,
        value=3.0,
        step=0.01
    )

    sks = st.number_input(
        "Total SKS",
        min_value=0,
        max_value=200,
        value=144,
        step=1
    )

    tombol = st.form_submit_button("Prediksi Kelulusan")


# 5. Logika ketika tombol ditekan
if tombol:

    # Prediksi menggunakan model
    hasil = model.predict([[ipk, sks]])

    # Tampilkan Output
    st.divider()
    st.subheader("Hasil Prediksi")

    if hasil[0] == 1:
        st.success(
            f"Mahasiswa dengan IPK **{ipk}** dan SKS **{sks}** "
            "diprediksi **LULUS TEPAT WAKTU**."
        )

    else:
        st.error(
            f"Mahasiswa dengan IPK **{ipk}** dan SKS **{sks}** "
            "diprediksi **TIDAK LULUS**."
        )
