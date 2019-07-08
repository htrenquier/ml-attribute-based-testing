from __future__ import division
import cv2

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import math
import data_splitting as ds
import model_trainer as mt


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def predict_batch(model, images):
    """
    Returns class predictions for image batch
    :param model:
    :param images:
    :return: List[int] of predicted classes
    """
    if images:
        y_predicted = model.predict(images)
        predicted_classes = np.argmax(y_predicted, axis=1)
        return predicted_classes.tolist()
    else:
        return []


def predict_dataset(filenames, path, model, model_preprocess_function):
    """
    For predicting large amount of images (e.g. imagenet)
    :param filenames: file of filenames
    :param model_preprocess_function:
    :param path: path of test images
    :param model:
    :return: predictions
    """
    y_predicted = []
    batch_size = 32
    batch = []
    for filename in filenames:
        batch.append(preprocess(path+filename, model_preprocess_function))
        if len(batch) >= batch_size:
            y_predicted = y_predicted + model.predict(np.array(batch)).tolist()
            batch = []
    y_predicted = y_predicted + model.predict(np.array(batch)).tolist()
    return y_predicted


def preprocess(file_path, model_preprocess_function):
    """
    Image pre-processing for large dataset prediction
    :param file_path:
    :param model_preprocess_function:
    :return:
    """
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    x = model_preprocess_function(x)
    return x



