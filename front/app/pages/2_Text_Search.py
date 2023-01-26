import streamlit as st
import pandas as pd
import requests
import cv2

captions_df = pd.read_csv(".\data\captions.csv")
captions_df = captions_df.drop_duplicates(subset=['image'], keep='first').reset_index(drop=True)
filenames = captions_df['image'].tolist()
PATH_IMAGES = ".\data\Images\\"

def search_images():
    matches = requests.get(
        url="http://localhost:3020/query_to_image/",
        params={
            "query": query
        }
    ).json()
    matched_indices = matches['matches']
    scores = matches['scores']
    for score, inx in zip(scores, matched_indices):
        image = cv2.imread(PATH_IMAGES+filenames[inx])
        st.image(image, channels="BGR")
        st.caption(str(score))

st.markdown("# Search Images based on the query")
st.sidebar.markdown("# Search Images")
query = st.text_input("Describe an image you are looking for!", value="", max_chars=256)
if st.button("Search images"):
    matches = requests.get(
        url="http://localhost:3020/query_to_image/",
        params={
            "query": query
        }
    ).json()
    matched_indices = matches['matches']
    scores = matches['scores']
    for score, inx in zip(scores, matched_indices):
        image = cv2.imread(PATH_IMAGES+filenames[inx])
        st.image(image, channels="BGR")
        st.caption(str(score))

