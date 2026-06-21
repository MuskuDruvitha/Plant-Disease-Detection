import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from disease_info import disease_info

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
.stApp {
    background-color: #f4faf4;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2E7D32;
    margin-bottom: 8px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# Page Heading
st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload a plant leaf image to identify diseases and receive management recommendations.</div>',
    unsafe_allow_html=True
)

st.divider()
# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_plant_disease_model.keras")

model = load_model()

# -----------------------------
# Load Class Indices
# -----------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping
class_names = {v: k for k, v in class_indices.items()}

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    # Format disease name
    display_name = predicted_class.replace("___", " - ").replace("_", " ")

    st.divider()

    # -----------------------------
    # Color-coded Result
    # -----------------------------
    if "healthy" in predicted_class.lower():
        st.success(f"✅ Healthy Leaf Detected\n\n**{display_name}**")
    else:
        st.error(f"🚨 Disease Detected\n\n**{display_name}**")

    # -----------------------------
    # Disease Information
    # -----------------------------
    if predicted_class in disease_info:

        info = disease_info[predicted_class]

        st.subheader("Description")
        st.write(info.get("description", "Not available."))

        st.subheader("Symptoms")
        st.write(info.get("symptoms", "Not available."))

        st.subheader("Prevention")
        st.write(info.get("prevention", "Not available."))

        st.subheader("Management")
        st.write(info.get("management", "Not available."))

    else:
        st.warning("No additional information available for this disease.")