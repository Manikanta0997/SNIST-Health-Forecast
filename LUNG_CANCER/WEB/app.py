import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

st.set_page_config(
    page_title="HASH-IT-OUT",
    page_icon="👋",
)
st.title("VITASCAN-HASH-IT-OUT")
st.sidebar.success("Select a page above.")
st.title("Lung cancer")
st.write('Lung cancer is the leading cause of cancer death worldwide, accounting for 1.59 million deaths in 2018. The majority of lung cancer cases are attributed to smoking, but exposure to air pollution is also a risk factor. A new study has found that air pollution may be linked to an increased risk of lung cancer, even in nonsmokers.')

st.write('The study, which was published in the journal Nature Medicine, looked at data from over 462,000 people in China who were followed for an average of six years. The participants were divided into two groups: those who lived in areas with high levels of air pollution and those who lived in areas with low levels of air pollution.')

st.write('The researchers found that the people in the high-pollution group were more likely to develop lung cancer than those in the low-pollution group. They also found that the risk was higher in nonsmokers than smokers, and that the risk increased with age.')

st.write('While this study does not prove that air pollution causes lung cancer, it does suggest that there may be a link between the two. More research is needed to confirm these findings and to determine what effect different types and levels of air pollution may have on lung cancer risk')
