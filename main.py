from google_drive_downloader import GoogleDriveDownloader as gdd

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Imports from repo main file

import json
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
import yolo
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image

from yolo.frontend import create_yolo

# 1. create yolo instance
yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)

# 2. load pretrained weighted file
# Pretrained weight file is at https://drive.google.com/drive/folders/1Lg3eAPC39G9GwVTCH3XzF73Eok-N-dER
# https://drive.google.com/file/d/1NmpmQB5Zmkzn306IN65AFXrzX6ITuGE_/view?usp=sharing

print("Downloading pretrained weights")

gdd.download_file_from_google_drive(file_id="1NmpmQB5Zmkzn306IN65AFXrzX6ITuGE_", dest_path='./weights.h5')
DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "weights.h5")
yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)


#####################################################################################################################################

def get_predictions(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    THRESHOLD = 0.3
    boxes, probs = yolo_detector.predict(img, THRESHOLD)
    pred = []
    for prob in probs:
        pred.append(np.argmax(prob))
    
    image = draw_scaled_boxes(img, boxes, probs, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    
    return image, pred

st.title('Reconciliation - Meter Reading Entered Manually Vs Photograph')
uploaded_file = st.file_uploader("Choose an image...",type=['png','jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    loading_text = st.text("Classifying...")
    transformed_image, label = get_predictions(image)
    st.write(label)
    loading_text.text('Prediction...')
    st.image(transformed_image, caption='Predictions made on the image.', use_column_width=True)
