🌱 Soil-Crop Prediction App
A machine learning-based web application that predicts the best crop to grow based on soil image analysis. This project uses image classification techniques and deep learning to assist farmers and agriculturists in decision-making.

📁 Project Structure
bash
Copy
Edit
soil-crop-app/
│--dataset                     # data to train the model
├── app.py                      # Streamlit web application
├── dataset_location_crops.csv # CSV mapping locations to suitable crops
├── soil_model.keras           # Saved Keras model (Keras format)
├── split_dataset.py           # Script to split dataset into train/test folders
├── train_model.py             # Model training script

📦 Download Model and Dataset
To keep the repository lightweight, the trained model and dataset have been provided separately:

🔗 Download model_and_dataset.zip :https://www.dropbox.com/scl/fi/biou54lme5ewvcdeypp9b/model_and_dataset.zip?rlkey=mdh6pqaarii00bhi1pe5fwyb2&st=v5q6nyub&dl=0   
(Contains: soil_model.keras and dataset/ folder)

After downloading:

Unzip the file in the project root folder:

bash
Copy
Edit
unzip model_and_dataset.zip
Make sure these two appear in the root directory:

Copy
Edit
├── soil_model.keras
└── dataset/
🧠 Why separate?
GitHub has file size limits (~100MB).

Using Releases or cloud links ensures faster repository cloning.

Keeps code and heavy assets modular.


🚀 Features
Upload a soil image to receive crop recommendations.

Uses a deep learning model trained on categorized soil images.

User-friendly Streamlit interface.

Extensible for location-based and soil health data.

🧠 Model Training
Model is trained using image classification techniques (likely CNNs) via train_model.py. The training dataset is split using split_dataset.py.

Key Tools/Libraries:

TensorFlow / Keras

OpenCV (for image processing)

NumPy, Pandas

Streamlit (for frontend)

📊 Dataset
Images: Soil images organized by class/folder.

CSV (dataset_location_crops.csv): Links locations or soil types to recommended crops.

Install dependencies:


pip install -r requirements.txt
Run the web app:

streamlit run app.py


🧑‍💻 Author
Anunay Kumar
    Gmail: anunayit@gmail.com
📫 LinkedIn:https://www.linkedin.com/in/anunay-kumar-151578284
🎥 YouTube: Anunay Tech

