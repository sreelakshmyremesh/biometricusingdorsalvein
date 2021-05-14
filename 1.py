# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:07:27 2021

@author: USER
"""

import scipy.ndimage
import collections
import cv2
import pandas as pd
import numpy as np
import math
import time
from math import sqrt
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
import glob
from skimage.morphology import skeletonize as skelt
from skimage.morphology import thin
from skimage.morphology import covex_hull_image
from skimage.morphology import *
import numpy as np
import csv
def skeletonize(image_input):
    image=np.zeros_like(image_input)
    image[image_input==0]=1.0
    ouput=np.zeros_like(image_input)
    skeletone=skelt(image)
    output[skelton]=255
    cv2.bitwise_not(ouput,output)
    return output
def thinning_morph(image, kernal):
    """ Thinning image using morphological operatiosn
    :param image: 2d array uint8
    :param kernal:3x3 2d array unint8
    :return: thin images"""
    thinning_image=np.zeros_like(image)
    img=image.copy()