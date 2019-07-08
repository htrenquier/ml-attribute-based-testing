import math
import numpy as np


def confidence(prediction):
    """
    Metric to evaluate the confidence of a model's prediction of an image's class.
    :param prediction: List[float] per-class probability of an image to belong to the class
    :return: Difference between guessed class probability and the mean of other probabilities
    """
    m = np.max(prediction)
    return m - (sum(prediction) - m) / (len(prediction) - 1)


def prediction_rating(prediction, true_class):
    """
    Metric to evaluate the prediction of a model's prediction of an image's class.
    :param prediction: List[float] per-class probability of an image to belong to the class
    :param true_class: The class the image actually belongs to.
    :return:
    """
    p_true = prediction[true_class]
    prediction = np.delete(prediction, true_class)
    p_max, p_min = np.max(prediction), np.min(prediction)
    if p_max == p_min:
        assert p_max < 0.01
        return 1
    x = (1 + p_true - p_max) / (p_max - p_min)
    return math.atan(x)*2/math.pi


def prediction_ratings(predictions, true_classes):
    return [prediction_rating(predictions[i], true_classes[i]) for i in xrange(len(predictions))]


def confidences(predictions):
    return [confidence(p) for p in predictions]


def accuracy(predicted_classes, true_classes):
    """
    Computes accuracy of a model based on the predictions /predicted_classes/.
    :param predicted_classes: List[int] Classes guessed by the model
    :param true_classes: List[int] Ground-truth
    :return: float Accuracy of the input predictions
    """
    nz = np.count_nonzero(np.subtract(predicted_classes, true_classes))
    acc = (len(true_classes) - nz) / len(true_classes)
    # print('Test Accuracy = ' + str(acc))
    return acc


# ====== # Sorting # ====== #


def sort_by_correctness(predictions, true_classes, orig_images):
    """
    Separates a test dataset into correctly guessed images and incorrectly guessed images.
    :param predictions:
    :param true_classes:
    :param orig_images:
    :return:
    """
    correct_images = []
    incorrect_images = []
    for i in xrange(len(predictions)):
        if predictions[i] == true_classes[i]:
            correct_images.append(orig_images[i])
        else:
            incorrect_images.append(orig_images[i])
    return correct_images, incorrect_images


def sort_by_confidence(confidences, number_elements=None):
    """
    Crescent sort
    :param confidences: List of confidences
    :param number_elements: How many elements to return
    :return: Two lists of indexes for high and low confidences.
    """
    if number_elements is None or number_elements > len(confidences)//2:
        number_elements = len(confidences)//2
    sorted_args = np.argsort(confidences)
    # return high_confidence, low_confidence
    return sorted_args[-number_elements:], sorted_args[:number_elements]
