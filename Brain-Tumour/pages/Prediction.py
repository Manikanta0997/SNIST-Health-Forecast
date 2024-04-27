import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('Brain-Tumour/Brain_tu.hdf5')
	return model
model = load_model()
st.title('Flower Classifier')
