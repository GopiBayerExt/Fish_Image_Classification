import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model (adjust path)
model = load_model("model/MobileNet_Model.keras")

# Label mapping
index_to_label = {
    0: "Black Sea Sprat",
    1: "Gilt Head Bream",
    2: "Horse Mackerel",
    3: "Red Mullet",
    4: "Red Sea Bream",
    5: "Sea Bass",
    6: "Shrimp",
    7: "Striped Red Mullet",
    8: "Trout"
}

# Set Streamlit page config
st.set_page_config(page_title="Fish Species Classifier", layout="centered")

# Inject internal CSS for padding and max width
st.markdown("""
    <style>
    .stMainBlockContainer {
        padding: 6rem 2rem 1rem;
        max-width: 800px;
    }
    h1 {
        display: flex;
        justify-content: center;        
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: 450px;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for page control
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'upload' not in st.session_state:
    st.session_state['upload'] = None

st.title("🐟 Fish Species Classifier")

class Pages:
    def home(self):
        with col1:
            st.image('images/fish.png') 

        with col2:
            st.markdown("""
                This app classifies fish species using a deep learning model trained on fish images.
                Upload a fish image to get instant predictions with confidence scores.
                """, unsafe_allow_html=True)

            st.session_state['upload'] = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

            if st.button("🔍 Predict"):
                if st.session_state['upload'] is not None:
                    st.session_state['page'] = 'result'
                    st.rerun()
                else:
                    st.warning("Please upload an image before clicking Predict.")

    def result(self):
        if st.session_state['upload'] is not None:
            with col1:
                img = Image.open(st.session_state['upload']).convert("RGB")
                st.image(img, caption="Uploaded Image", use_container_width=True)

            # Preprocess 
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            predictions = model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            predicted_label = index_to_label[predicted_index]
            confidence = predictions[predicted_index] * 100

            with col2:
                st.markdown(f"### Prediction:")
                st.markdown(f"### `{predicted_label}`")
                st.markdown(f"**Confidence:** {confidence:.2f}%")

                if st.button('🔙 Back to Home'):
                    st.session_state['page'] = 'home'
                    st.rerun()
        else:
            st.session_state['page'] = 'home'
            st.rerun()

page = Pages()
with st.container(border=True):
    col1, col2 = st.columns(2, gap='medium', vertical_alignment='center')

    if st.session_state['page'] == 'home':
        page.home()
    elif st.session_state['page'] == 'result':
        page.result()