import streamlit as st
import pickle
import pandas as pd
import lzma
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from io import StringIO
import numpy as np

st.set_page_config(page_title="Satellite Anomaly Detection", page_icon=None, layout="wide",
                   initial_sidebar_state="expanded", menu_items={"Get help": None, "Report a Bug": None, "About": None})

best_model = pickle.load(lzma.open('model.pickle', 'rb'))

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    default_image_size = tuple((224,224))
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR")
    image = cv2.resize(img, default_image_size)
    prediction = best_model.predict(np.array([img_to_array(image)]))
    if prediction > 0.5:
        prediction = 1
    else:
        prediction = 0

    if prediction == 0:
        st.write("It is cloudy!")
    else:
        st.write("It is clear!")