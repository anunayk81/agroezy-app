import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests

# Weather API key
WEATHER_API_KEY = "369ff24c15031a477ece5f447383442d"

# Load trained model
model = tf.keras.models.load_model("soil_model.keras")

# Load location-crops CSV
location_crops_df = pd.read_csv("location_crops.csv")

# Class names
class_names = ['Alluvial', 'Arid', 'Black', 'Laterite', 'Mountain', 'Red', 'Yellow']

# Soil to crops mapping
soil_crops = {
    # … same as before …
}

def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp'], data['main']['humidity'], data['weather'][0]['description']
    else:
        return None, None, None

# 🌱 Improved styling
st.markdown("""
<style>
body {
    background-color: #eaf4ea;
}
.stApp {
    background-color: #eaf4ea;
    color: #333333;
}
h1 {
    color: #2e7d32;
    text-align: center;
}
p {
    text-align: center;
    color: #4d4d4d;
}
.stTextInput>div>div>input, .stTextArea>div>textarea {
    background-color: #ffffff;
    color: #000000;
    border: 1px solid #cccccc;
    border-radius: 5px;
    padding: 8px;
}
.stButton>button {
    background-color: #4caf50;
    color: white;
    border-radius: 5px;
    padding: 8px 16px;
    font-weight: bold;
}
.stFileUploader {
    background-color: #ffffff;
    border-radius: 5px;
    padding: 8px;
}
.stAlert {
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌾 AgroEzy: Soil + Weather + Location Crop Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p>What drove us towards taking up farming was our love for plants.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])
city = st.text_input("City:").strip().title()
state = st.text_input("State:").strip().title()
ph_value = st.text_input("Soil pH (optional):")
nutrients = st.text_input("Soil Nutrients (optional):")

if uploaded_file and city:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_soil = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"✅ Predicted Soil: {predicted_soil} ({confidence:.2f}%)")
    st.info(f"📍 Location: {city}, {state}")

    temp, humidity, weather_desc = get_weather(city, WEATHER_API_KEY)
    if temp:
        st.info(f"🌦 Weather: {weather_desc}, {temp}°C, Humidity: {humidity}%")

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
            st.info(f"🧪 Soil pH: {ph} ({ph_type})")
        except:
            st.warning("⚠ Invalid pH value")

    if nutrients:
        st.info(f"🧪 Nutrient Status: {nutrients}")

    if recommended:
        st.success("🌾 Recommended Crops: " + ", ".join(sorted(recommended)))
    else:
        st.warning("No recommended crops found for this combination.")
else:
    st.info("📌 Please upload a soil image and enter city/state.")
