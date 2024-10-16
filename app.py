import streamlit as st
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch

# Set page title and layout
st.set_page_config(page_title="Apple Quality Assessment", layout="wide")


# Function to load pre-trained model and feature extractor
@st.cache_resource
def load_model():
    model_name = "bazaar/cv_apple_leaf_disease_detection"  # Updated model name
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None  # Return None to avoid further errors


# Load model and feature extractor
model, feature_extractor = load_model()


# Function to predict the apple condition
def predict_image(image):
    # Ensure the image is in RGB format
    try:
        # Convert the image to RGB (removing alpha channel if it exists)
        if image.mode == "RGBA":
            image = image.convert("RGB")  # Convert RGBA to RGB
        elif image.mode != "RGB":
            image = image.convert("RGB")  # Convert any other format to RGB
    except Exception as e:
        st.error(f"Error converting image to RGB: {e}")
        return "Error"

    # Preprocess the image
    try:
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # Define class names based on the model's output
        class_names = model.config.id2label  # Fetch the dynamic class labels

        # Return the predicted class name
        return class_names[predicted_class_idx]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"


# Design the UI
st.title("🍎 Apple Quality Assessment")
st.markdown("""
Upload an image of an apple leaf, and the app will assess its condition using a pre-trained machine learning model.
""")

# File uploader for image input
uploaded_image = st.file_uploader("Upload an apple leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and preprocess the image
    image = Image.open(uploaded_image)

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])  # Adjust the ratios as needed

    with col1:
        # Display the uploaded image in a smaller size
        st.image(image, caption="Uploaded Apple Leaf Image", use_column_width=True)

    with col2:
        # Predict and show the result
        with st.spinner("Analyzing..."):
            result = predict_image(image)
            st.success(f"The apple leaf condition is: {result}")

# Sidebar for additional info or options
st.sidebar.header("About this App")
st.sidebar.write("""
This app uses the `bazaar/cv_apple_leaf_disease_detection` pre-trained model from Hugging Face to classify the condition of apple leaves. 
It can detect various conditions such as:
- Alternaria leaf spot
- Brown spot
- Frogeye leaf spot
- Grey spot
- Healthy
- Mosaic
- Powdery mildew
- Rust
- Scab
""")

