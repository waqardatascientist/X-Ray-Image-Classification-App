import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('my_covid19_model.h5')

# Function to preprocess the image and make predictions
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize image to (256, 256)
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("COVID-19 X-Ray Image Classification")
st.write("Upload an X-ray image, and the model will predict whether it's COVID-19 positive or negative.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray Image', use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Display the result
    if prediction[0] > 0.5:
        st.write("Prediction: COVID-19 Positive")
    else:
        st.write("Prediction: COVID-19 Negative")
