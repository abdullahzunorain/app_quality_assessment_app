import streamlit as st
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch

# Set page title and layout
st.set_page_config(page_title="Apple Quality Assessment", layout="wide")

# Function to load pre-trained model and feature extractor
@st.cache_resource
def load_model():
    model_name = "bazaar/cv_apple_leaf_disease_detection"  # Using the Hugging Face model you mentioned
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

# Load model and feature extractor
model, feature_extractor = load_model()

# Function to predict if apple is healthy or diseased
def predict_image(image):
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Define class names based on model's output
    class_names = ["Healthy", "Diseased"]  # Adjust these class labels if needed
    
    return class_names[predicted_class_idx]

# Design the UI
st.title("🍎 Apple Quality Assessment")
st.markdown("""
Upload an image of an apple, and the app will assess whether the apple is **healthy** or **diseased** using the pre-trained machine learning model.
""")

# File uploader for image input
uploaded_image = st.file_uploader("Upload an apple image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Apple Image", use_column_width=True)

    # Open and preprocess the image
    image = Image.open(uploaded_image)
    
    # Predict and show the result
    with st.spinner("Analyzing..."):
        result = predict_image(image)
        if result == "Healthy":
            st.success(f"The apple is {result} 🍏")
        else:
            st.error(f"The apple is {result} 🍎")

# Sidebar for additional info or options
st.sidebar.header("About this App")
st.sidebar.write("""
This app uses the `bazaar/cv_apple_leaf_disease_detection` pre-trained model from Hugging Face to classify whether an apple is healthy or diseased. 
It is designed to provide fast and accurate results.
""")
