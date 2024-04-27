import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import skimage.io
import skimage.segmentation
import copy
import sklearn
from sklearn.metrics.pairwise import cosine_distances
from sklearn.linear_model import LinearRegression

st.set_option('deprecation.showfileUploaderEncoding', False)

def load_model():
    model = tf.keras.models.load_model('Brain-Tumour/brain_final.hdf5')
    return model

model = load_model()
st.title('Brain Tumour Prediction')

def predict_class(image, model):
    lime_img = skimage.transform.resize(image, (150, 150))
    image.shape
    resized_image = cv2.resize(image, (150, 150))
    
    resized_image.shape
    resized_image.size
    reshaped_image = np.reshape(resized_image, (1, 150, 150, 3))
    
    predictions = model.predict(reshaped_image)
    pred = predictions[0][0]  
    clas = 0
    if pred > 0.5:
        pred = "HEALTHY"
        clas = 1
    else:
        pred = "BRAIN_TUMOR"
        clas = 0
    return [pred, lime_img, clas]

def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png", "jpeg", "tif"])

if file is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')

    test_image = Image.open(file)

    st.image(test_image, caption="Input Image", width=400)
    
    pred, img, clas = predict_class(np.asarray(test_image), model)

    result = pred

    output = 'The image is a ' + result

    slot.text('Done')
    st.success(output)
    
    if clas == 0:
        superpixels = skimage.segmentation.quickshift(img, kernel_size=4, max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]
        
        st.image(skimage.segmentation.mark_boundaries(img / 2 + 0.5, superpixels), width=400)
        
        num_perturb = 150
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
        
        st.image(perturb_image(img / 2 + 0.5, perturbations[0], superpixels), width=400)
        Xi = img
        
        predictions = []
        for pert in perturbations:
            perturbed_img = perturb_image(Xi, pert, superpixels)
            pred = model.predict(perturbed_img[np.newaxis, :, :, :])
            predictions.append(pred)

        predictions = np.array(predictions)
        
        original_image = np.ones(num_superpixels)[np.newaxis, :]
        distances = cosine_distances(perturbations, original_image)
        
        if clas == 0:
            clas = [0, 1]
        else:
            clas = [1, 0]    
        
        top_pred_classes = clas
        
        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        
        weights = np.ravel(weights)
        
        class_to_explain = top_pred_classes[0]
        simpler_model = LinearRegression()
        simpler_model.fit(X=perturbations, y=predictions[:, :, class_to_explain], sample_weight=weights)
        coeff = simpler_model.coef_[0]

        num_top_features = 1
        
        top_features = np.argsort(coeff)[-num_top_features:]
        
        mask = np.zeros(num_superpixels)
        mask[top_features] = True
        st.image(perturb_image(Xi / 2 + 0.5, mask, superpixels), width=400)

