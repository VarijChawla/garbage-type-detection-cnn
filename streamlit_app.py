import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("../models/waste_classifier.h5")
IMG_SIZE = 128
classes = ['Recyclable', 'Biodegradable', 'Non-Recyclable']

st.title("♻️ Smart Waste Classification")
st.markdown("Upload an image to classify its waste type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ✅ Convert to RGB
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_resized)
    result = classes[np.argmax(pred)]
    st.success(f"Prediction: **{result}**")
