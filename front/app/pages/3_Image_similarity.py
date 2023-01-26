import streamlit as st
import numpy as np
import cv2
from io import StringIO
import requests
import pandas as pd

captions_df = pd.read_csv(".\data\captions.csv")
captions_df = captions_df.drop_duplicates(subset=['image'], keep='first').reset_index(drop=True)
filenames = captions_df['image'].tolist()
PATH_IMAGES = ".\data\Images\\"

st.markdown("# Find similar images")
st.sidebar.markdown("# Image similarity")

uploaded_image = st.file_uploader("Upload an image")
if uploaded_image is not None:
    string_data = uploaded_image.read()
    img = np.frombuffer(string_data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.write("Uploaded image: ")
    st.image(img)
    file = {'file': string_data}
    response = requests.post(
        url="http://localhost:3020/similar_images/",
        files=file
        ).json()
    matched_indices = response['matches']
    scores = response['scores']
    st.title("Search results:")
    for score, inx in zip(scores, matched_indices):
        image = cv2.imread(PATH_IMAGES+filenames[inx])
        st.image(image, channels="BGR")
        st.caption(str(score))