import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from PIL import Image
import cv2 
import matplotlib.pyplot as plt


def imfilter(image, kernel):
    # filter the img to get high or low frequncies
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image


def gen_hybrid_img(image1, image2, cutoff_frequency, weight_factor):
    # Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
    # Combine them to create 'hybrid_image'.

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z / (2*s*s)) / sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)


    low_frequencies = imfilter(image1, kernel)
    high_frequencies = image2 - imfilter(image2, kernel)

    hybrid_image = low_frequencies + weight_factor * high_frequencies 

    hybrid_image = np.clip(hybrid_image, 0, 1)

    return low_frequencies, high_frequencies, hybrid_image