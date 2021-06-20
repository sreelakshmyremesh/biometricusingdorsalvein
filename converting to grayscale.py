# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 19:39:36 2021

@author: USER
"""

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('img1.png')

R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(imgGray, cmap='gray')
plt.show()
        