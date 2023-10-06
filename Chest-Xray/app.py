import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)
st.title("VITASCAN_SMART_O_THON")
st.sidebar.success("Select a page above.")

st.title('Pneumonia Detection')
st.write('Pneumonia is a serious respiratory infection caused by bacteria, viruses, fungi, or parasites that can be life-threatening, particularly for vulnerable populations. Detecting pneumonia is crucial for several reasons. Firstly, early diagnosis allows for prompt treatment, reducing complications and improving patient outcomes. Accurate detection is essential in differentiating pneumonia from other respiratory conditions, ensuring proper management. By determining the specific type and severity of pneumonia, detection helps healthcare providers tailor the treatment approach accordingly. Timely identification and treatment also minimize the risk of complications such as pleural effusion and respiratory failure. Additionally, accurate detection plays a vital role in public health surveillance by identifying outbreaks and guiding preventive measures to control the spread of pneumonia. Lastly, appropriate detection of bacterial pneumonia supports antibiotic stewardship efforts, which are crucial in combating antibiotic resistance. Diagnostic methods such as physical examination, chest X-rays, blood tests, sputum culture, and molecular tests aid in the timely and accurate detection of pneumonia. Overall, timely and accurate detection of pneumonia is of paramount importance as it ensures proper treatment, prevents complications, and safeguards public health.')
st.set_option('deprecation.showfileUploaderEncoding', False)

