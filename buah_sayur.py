import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Buah & Sayur",
    page_icon="🍓",
    layout="wide"
)

# Load model (pakai try-except agar tidak error saat deploy)
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("model_buah_sayur.h5")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_cnn_model()

# Daftar nama kelas
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Sidebar info
st.sidebar.image("https://img.icons8.com/color/96/vegetarian-food.png", width=100)
st.sidebar.title("🍽️ Info Aplikasi")
st.sidebar.markdown("""
Aplikasi ini menggunakan model **CNN** untuk mengenali buah dan sayur.  
Upload gambar dan lihat 3 hasil prediksi teratas lengkap dengan skor kepercayaannya.  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("📍 Dibuat dengan ❤️ oleh Imam Husen")

# Header halaman utama
st.title("🍎 Aplikasi Klasifikasi Buah & Sayur")
st.markdown("Upload gambar buah atau sayur dan biarkan model memprediksi dengan cerdas! 🤖🥬🍍")

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload Gambar di Sini (jpg/png/jpeg):", type=["jpg", "jpeg", "png"])

# Proses jika file terupload dan model tersedia
if uploaded_file is not None and model:
    col1, col2 = st.columns([1, 2])

    # Tampilkan gambar
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📷 Gambar yang Diunggah", use_container_width=True)

    # Proses prediksi
    with col2:
        st.markdown("### 🔍 Top-3 Prediksi")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        

        pred = model.predict(img_array)
        top3 = np.argsort(pred[0])[-3:][::-1]

        for i in top3:
            st.markdown(f"""
            <div style="font-size:20px;padding:5px 0;">
                ✅ <b>{class_names[i].capitalize()}</b> — 
                <span style="color:#FF4B4B">{pred[0][i]*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

elif uploaded_file is not None and not model:
    st.warning("Model belum berhasil dimuat.")
