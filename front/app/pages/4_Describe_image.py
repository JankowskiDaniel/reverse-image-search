import streamlit as st
import pandas as pd
import numpy as np
import cv2
import requests

st.markdown("# Describe uploaded image")
st.sidebar.markdown("# Image Descriptor")


captions_df = pd.read_csv(".\data\captions.csv")
# captions_df = captions_df.drop_duplicates(subset=['image'], keep='first').reset_index(drop=True)
CAPTIONS = captions_df['caption'].tolist()
# PATH_IMAGES = ".\data\Images\\"

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
        url="http://localhost:3020/describe_image/",
        files=file
        ).json()
    matched_indices = response['matches']
    scores = response['scores']
    st.title("Proposed image description:")
    for score, inx in zip(scores, matched_indices):
        st.write(CAPTIONS[inx])
        st.caption(str(score))


