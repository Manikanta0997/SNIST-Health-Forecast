import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.set_page_config(
    page_title="HASH-IT-OUT(SNIST)",
    page_icon="👋",
)
st.header('VITASCAN-(HASH-IT-OUT)')
st.title('Brain tumour detection')
st.write('A brain tumor is an abnormal growth of cells in the brain. It can originate from brain cells or cells that have spread from other parts of the body to the brain. Brain tumors can be either benign (non-cancerous) or malignant (cancerous).')
st.write("We are going to predict 3 possible tumour that are 1)Glioma Tumor 2)Meningioma Tumor 3)No Tumor 4)Pituitary Tumor")
st.write('1)Glioma Tumor: A glioma is a type of brain tumor that originates in the glial cells, which are the supportive cells of the central nervous system. Common symptoms may include headaches, seizures, cognitive changes, personality changes, weakness, and sensory disturbances. The diagnosis of gliomas typically involves a combination of imaging tests, such as magnetic resonance imaging (MRI) and computed tomography (CT) scans.')
st.write('2)Meningioma Tumor: A meningioma is a type of tumor that arises from the meninges, which are the protective membranes that cover the brain and spinal cord. larger or growing meningiomas can cause symptoms by compressing nearby brain tissue. The diagnosis of a meningioma typically involves a combination of imaging tests, such as magnetic resonance imaging (MRI) and computed tomography (CT) scans.')
st.write('3)Pituitary Tumor: The pituitary gland is a small, pea-sized gland located at the base of the brain and is often referred to as the "master gland" because it produces and regulates several hormones that control various bodily functions. The symptoms of a pituitary tumor include headaches, vision problems, hormonal imbalances, fatigue, menstrual irregularities, sexual dysfunction, and changes in weight or appetite. Diagnosis of a pituitary tumor typically involves a combination of imaging tests, such as magnetic resonance imaging (MRI).')
st.set_option('deprecation.showfileUploaderEncoding', False)

