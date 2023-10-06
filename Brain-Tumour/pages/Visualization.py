import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Visualization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title('Brain Tumour MRI Visualization')

st.header('1.NORMAL BRAIN MRI:') 
img = Image.open('image.png')
st.image(img)

st.header('2.Meningioma Tumor')
img = Image.open('image_1.png')
st.image(img)
st.header('3.Glioma Tumor')
img = Image.open('image_2.png')
st.image(img)
img = Image.open('image_3.png')
st.header('4.Pituitary Tumor')
st.image(img)



