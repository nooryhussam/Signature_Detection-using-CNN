# Import necessary libraries
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load the trained signature verification model
model = load_model("E:\Python\saved_modelsignature_2.keras")

# Function to preprocess the uploaded signature image
def preprocess_signature(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ValueError("Could not read the image.")
    img = cv2.resize(img, (128, 128))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)
    return img

# Function to interpret the model prediction
def interpret(score):
    return "Original" if score > 0.5 else "Forged"

# Streamlit UI: Page title
st.markdown("""
    <h1 style='text-align: center;'>✍️ Signature Detection</h1>
""", unsafe_allow_html=True)

# Instruction message
st.markdown("""
<div style="padding-top: 30px; text-align: left; margin-left: 10px; padding-bottom:20px;">
    <strong>Upload a Signature Image to Verify.</strong>
</div>
""", unsafe_allow_html=True)

# File uploader
signature_file = st.file_uploader("Upload Test Signature", type=["png", "jpg", "jpeg"])

# If file uploaded
if signature_file:
    signature_file.seek(0)
    img_bytes = np.asarray(bytearray(signature_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img_resized = cv2.resize(img, (600, 400))
        st.image(img_resized, channels="GRAY", use_container_width=True)

    try:
        signature_file.seek(0)
        processed_img = preprocess_signature(signature_file)
        prediction = model.predict(processed_img)
        score = prediction[0][0]
        result = interpret(score)

        # Adjust colors for Original vs Forged
        bg_color = "#d4edda" if result == "Original" else "#f8d7da"   
        text_color = "#155724" if result == "Original" else "#721c24"  
        border_color = "#c3e6cb" if result == "Original" else "#f5c6cb"

        # Display styled result box
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='
            padding: 1rem;
            border-radius: 8px;
            background-color: {bg_color};
            border-left: 6px solid {border_color};
            font-size: 1.2rem;
            font-weight: bold;
            color: {text_color};'>
            Prediction: {result} (Score: {score:.2f})
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing the signature: {e}")
