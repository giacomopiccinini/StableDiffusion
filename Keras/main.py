import keras_cv
import cv2
from tensorflow import keras

import logging 
logging.getLogger('tensorflow').setLevel(logging.ERROR) 

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
h5py._conv.logger.setLevel("ERROR")

keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion(img_width=128, img_height=128)

images = model.text_to_image(
    "a cute magical flying dog, fantasy art, "
    "golden color, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure",
    batch_size=1,
    num_steps=1
)