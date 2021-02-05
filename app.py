import json
import streamlit as st

from copy import deepcopy
import json
import random

import time
from pathlib import Path
import os
import numpy
from PIL import Image, ImageDraw

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras import backend as K
import keras

# keras.models.load_model
# from keras.models import Model, load_model
# import tensorflow as tf

# from models import unet, unet_mobile


def show_image_mask(im, mask, alpha=0.3):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(im)
    plt.imshow(mask, alpha=alpha)
    return fig

def cut_512(im):
    im_crop = []
    for i in range(0, 512, 128):
        for j in range(0, 512, 128):
            im_crop.append(im[i:i + 128, j:j + 128, :])
    return im_crop

def recut_512(im_crop, mask=False):
    if mask:
        im = np.zeros((512, 512, 3))
    else:
        im = np.zeros((512, 512, 3), "int")
    for i in range(16):
        x = i // 4
        y = i % 4
        im[x * 128: (x + 1) * 128, y * 128: (y + 1) * 128, :] = im_crop[i]
    return im

def load_models_list():
    with open('models/models.json', 'r', encoding="utf-8") as fp:
        models = json.load(fp)
    return models

MODELS = st.cache(load_models_list, allow_output_mutation=True)()

def load_image(image_file):
    img = Image.open(image_file)
    return img

def on_click_execute(image, model, tr_opt):
    """Click button
    """
    if st.sidebar.button("Сегментировать"):
        if image is not None:
            im_crop = cut_512(image)
            mask_image = []

            for im in im_crop:
                prob_pred = model.predict(im.reshape(1, 128, 128, 3) / 255.0)
                mask_image.append((prob_pred > tr_opt) * 1)
            
            mask = recut_512(mask_image, True)
            return mask
        return None

@st.cache(allow_output_mutation=True)
def load_model_(type_target, model_file):
    print("loading model...")
    model = keras.models.load_model(f"models/{model_file}", compile=False)
    print("model loaded!")
    opt_tr = MODELS[type_target][model_file]["threshold"]

    return model, opt_tr


def main():
    """
    Функция - построение веб-формы
    """
    st.markdown("# Сегментация")

    type_target = st.sidebar.selectbox("Тип подстилающей поверхности:", list(MODELS.keys()))
    
    avaliable_models = list(MODELS[type_target].keys())
    model_file = st.sidebar.selectbox("Доступные модели:", avaliable_models)
    avaliable_model, opt_tr = load_model_(type_target, model_file)

    uploaded_file = st.sidebar.file_uploader("Выберите изображение для анализа", type=['png', 'jpeg'])
    image = None
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        image = image.convert('RGB')
        image = np.array(image)

        if image.shape == (512, 512, 3):
            st.markdown("### Выбранное изображение")
            st.image(image, width=350, height=350)
        else:
            st.markdown("Должно быть разрешение 512x512. Но мы совершенствуемся!")
    mask = on_click_execute(image, avaliable_model, opt_tr)
    if mask is not None:
        fig = show_image_mask(image, mask)
        st.markdown("### Результат сегментации")
        st.image(np.clip(image + 255 * mask.astype(int), 0, 255), width=350, height=350)

if __name__ == "__main__":
    main()