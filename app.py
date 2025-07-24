import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests

# Weather API key
WEATHER_API_KEY = "369ff24c15031a477ece5f447383442d"

# Load model
model = tf.keras.models.load_model("soil_model.keras")

# Load location-crops CSV
location_crops_df = pd.read_csv("location_crops.csv")

# Soil classes
class_names = ['Alluvial', 'Arid', 'Black', 'Laterite', 'Mountain', 'Red', 'Yellow']

# Soil to crop mapping (use full data here)
soil_crops = {
    'Alluvial': ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Pulses'],
    'Arid': ['Bajra', 'Millets', 'Barley'],
    'Black': ['Cotton', 'Soybean', 'Groundnut', 'Rice', 'Banana', 'Flowers', 'Guava', 'Jamun', 'Jute', 'Lemon', 'Maize', 'Medicinal Plants'],
    'Laterite': ['Tea', 'Coffee', 'Cashew'],
    'Mountain': ['Tea', 'Apples', 'Barley'],
    'Red': ['Millets', 'Pulses', 'Groundnut'],
    'Yellow': ['Paddy', 'Maize', 'Potato']
}

# Weather API function
def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp'], data['main']['humidity'], data['weather'][0]['description']
    else:
        return None, None, None

# ---------- Streamlit App ----------
st.set_page_config(page_title="AgroEzy: Smart Crop Recommender", layout="centered")

# 🌱 Custom CSS styling
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #e0e0e0;
}
.stApp {
    background-color: #121212;
}
h1 {
    text-align: center;
    color: #90ee90;
    font-size: 36px;
}
p {
    text-align: center;
    color: #c0c0c0;
}
small {
    position: fixed;
    bottom: 5px;
    right: 10px;
    font-size: 10px;
    color: #444;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🌾 AgroEzy: Soil + Weather + Location Crop Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p>Empowering Farmers with AI, Soil & Weather 🌦️</p>", unsafe_allow_html=True)

# Upload input and info columns
uploaded_file = st.file_uploader("📷 Upload Soil Image", type=["jpg", "png", "jpeg"])

with st.container():
    col1, col2 = st.columns(2)
    ph_value = col1.text_input("🌿 Soil pH (optional)")
    nutrients = col2.text_input("🧬 Soil Nutrients (optional)")

# State and city selection with filtering
all_states = sorted(location_crops_df['state'].dropna().unique())
state = st.selectbox("📍 Select State", all_states)

filtered_cities = location_crops_df[location_crops_df['state'] == state]['city'].dropna().unique()
city = st.selectbox("🏙️ Select City", sorted(filtered_cities))

# Prediction & Results
if uploaded_file and city:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_soil = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # 💡 Improved result card
    st.markdown(f"""
    <div style="background-color:#1e2b1e; padding:20px; border-radius:10px;">
        <div style="display:flex; flex-wrap:wrap; justify-content:space-between;">
            <div style="color:#ffffff; font-size:20px; font-weight:bold;">
                🧬 Predicted Soil: <span style="color:#a2faa3;">{predicted_soil}</span>
            </div>
            <div style="color:#ffffff; font-size:20px;">
                📍 {city}, {state}
            </div>
        </div>
        <div style="margin-top:10px; font-size:18px; color:#ffffff;">
            Confidence: <span style="font-weight:bold; color:#a2faa3;">{confidence:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    temp, humidity, weather_desc = get_weather(city, WEATHER_API_KEY)
    if temp:
        st.markdown(f"""
        <div style="background-color:#10253f; padding:15px; border-radius:10px; color:#d0e6ff;">
            🌦️ <b>Weather:</b> {weather_desc}, {temp}°C, Humidity: {humidity}%
        </div>
        """, unsafe_allow_html=True)

    row = location_crops_df[
        (location_crops_df['city'].str.title() == city) &
        (location_crops_df['state'].str.title() == state)
    ]
    crops_location = []
    if not row.empty:
        crops_location = row.iloc[0]['crops'].split('|')

    crops_soil = soil_crops.get(predicted_soil, [])
    recommended = list(set(crops_location) & set(crops_soil))
    if not recommended:
        recommended = crops_location or crops_soil

    if ph_value:
        try:
            ph = float(ph_value)
            if ph < 6.0:
                ph_type = "Acidic"
            elif ph <= 7.5:
                ph_type = "Neutral"
            else:
                ph_type = "Alkaline"
            st.markdown(f"""
            <div style="background-color:#344c3c; padding:10px; border-radius:8px; color:#fff;">
            🧪 Soil pH: {ph} ({ph_type})
            </div>
            """, unsafe_allow_html=True)
        except:
            st.warning("⚠ Invalid pH value")

    if nutrients:
        st.markdown(f"""
        <div style="background-color:#344c3c; padding:10px; border-radius:8px; color:#fff;">
        🧬 Nutrient Status: {nutrients}
        </div>
        """, unsafe_allow_html=True)

    if recommended:
        st.markdown(f"""
        <div style="background-color:#1e3c2e; padding:20px; border-radius:10px; color:#d2ffd0;">
        ✅ <b>Recommended Crops:</b> {', '.join(sorted(recommended))}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No recommended crops found for this combination.")
else:
    st.info("📌 Please upload a soil image and select state/city.")

# Signature
st.markdown("<small>Built by Anunay Kumar</small>", unsafe_allow_html=True)


