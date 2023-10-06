import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

pipe = pickle.load(open('WEB/Lung_Cancer_Prediction_using_KNN10.pkl','rb'))
st.set_page_config(
    page_title="Lung_Cancer_Predictor",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title('Lung_Cancer_Predictor')


col1, col2 = st.columns(2)

with col1:
	Alcohol_use = st.number_input('Alcohol use:Range(1-8)')
with col2:
  	Dust_Allergy = st.number_input('Dust_Allergy:Range(1-8)')
col3, col4 = st.columns(2)
with col3:
	Genetic_Risk = st.number_input('Genetic Risk:Range(7-1)')
with col4:
	Balanced_Diet = st.number_input('Balanced Diet:Range(7-1)')
	
col5, col6, col7 = st.columns(3)
with col5:
	Obesity = st.number_input('Obesity:Range(7-1)')
with col6:
	Passive_Smoker = st.number_input('Passive Smoker:Range(8-1)')
with col7: 
	Coughing_of_Blood = st.number_input('Coughing of Blood:Range(9-1)')


if st.button('Predict'):
    input_df = pd.DataFrame(
     {'Alcohol use':[Alcohol_use],'Dust Allergy':[Dust_Allergy],'Genetic Risk':[Genetic_Risk],'Balanced Diet':[Balanced_Diet],'Obesity':[Obesity],'Passive Smoker':[Passive_Smoker],'Coughing of Blood':[Coughing_of_Blood]})
    result = pipe.predict(input_df)
    if int(result[0]) == 0:
    	result = "Low Chances Of Getting"
    elif int(result[0]) == 1:
    	result = "Medium Level Of Chances Of Getting"
    else:
    	result = "Very High Chances Of Getting"
    st.header("Predicted  - " + result)
    





st.header("Note: It is only for educational purpose")
