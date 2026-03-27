import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

model = load_model("../models/classification_model.keras", compile=False)

st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload MRI Image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    st.image(img, caption="Uploaded Image")
    st.write("Prediction:", class_names[np.argmax(pred)])