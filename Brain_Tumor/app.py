import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

st.title(" Brain Tumor Classifier")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    #  Reject suspicious images based on mode or size
    if img.mode not in ["L", "RGB"]:
        st.error(" This does not appear to be a valid grayscale or RGB MRI image.")
    elif img.size[0] < 200 or img.size[1] < 200:
        st.error(" Image is too small to be a valid brain MRI. Please upload a proper MRI scan.")
    else:
        #  Valid image → continue to preprocess, predict
        st.image(img, caption="Uploaded MRI Image", use_container_width=True)

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)


         # Choose which model to load (you can switch between them)
        model = load_model("model_tl.h5")  # or "cnn_model.h5"
        pred = model.predict(img_array)
        class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        confidence = np.max(pred)
        label = class_names[np.argmax(pred)]

        if confidence < 0.85:
            st.warning(" This may not be a valid brain MRI scan. Please upload a clearer medical image.")
        else:
            st.success(f" Prediction: {label}")
            st.markdown(f"###  Confidence: **{confidence * 100:.2f}%**")
