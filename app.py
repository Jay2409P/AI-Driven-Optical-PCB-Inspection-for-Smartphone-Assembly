# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# Load YOLOv8 model
model = YOLO('best.pt')  # Change path if needed

st.title("🔍 PCB Defect Detection - YOLOv8")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    # Predict
    results = model(tmp_path, conf=0.3)

    # Display result image
    result_img_path = results[0].save(filename='output.jpg')  # Save to disk
    st.image('output.jpg', caption="Detected Image", use_container_width=True)

    # Display detected class names
    st.subheader("📋 Detected Classes:")
    for box in results[0].boxes.data.tolist():
        class_id = int(box[5])
        class_name = results[0].names[class_id]
        st.write(f"• {class_name}")