import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

st.header('Are you sure want to leave this page?')


url = 'https://mellifluous-hamster-da427b.netlify.app/'


st.markdown(f'''
<a href={url}><button style="background-color:White;  height: 2em; width: 18em; font-size:30px;">Yes</button></a>
''',
unsafe_allow_html=True)

