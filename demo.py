import streamlit as st
from PIL import Image
import torch
import os

# Import your existing functions from test.py
from test import create_mobilenet_model, detect_objects, draw_boxes

# Load model
@st.cache_resource
def load_model(model_path, device):
    model = create_mobilenet_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# UI
st.title("Object Detection Demo")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.3)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = "quick_rcnn_model.pth"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found.")
        else:
            model = load_model(model_path, device)
            _, boxes, scores, labels = detect_objects(model, uploaded_file, device, confidence)
            result_image = draw_boxes(image.copy(), boxes, scores, labels)

            st.image(result_image, caption="Detection Results", use_container_width=True)
            st.success(f"✅ Detected {len(boxes)} objects with confidence ≥ {confidence}")
