# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:54:41 2021

@author: USER
"""

document_assembler=Documen 
import numpy as np

from PreProcessing_FeatureExtraction.connect_center import connect_centres
from PreProcessing_FeatureExtraction.detect_vein_center_assign_score import compute_vein_score
from PreProcessing_FeatureExtraction.label import binaries
from PreProcessing_FeatureExtraction.normalize import normalize_date
from PreProcessing_FeatureExtraction.preprocessing import remove_hair
from PreProcessing_FeatureExtraction.profile_curvature import compute_curvature


def vein_pattern(image, kernal_size, sigma):
    
#test
    
import cv2
import matplotlib.pyplot as plt
image_path='../sample dataset/input/s1/2017232_R_S.jpg'
image=cv2.imread(image_path,0)
processed_image=vein_pattern(image,6,8)
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(processed_Image)
plt.subtitle("Vein Pattern")
plt.tight_layout()
plt.savefig("vein_pattern_extracted.png")
plt.show()

def compute_curvature(image, sigma):
    
    #1. constructs the 2D gaussian filter "h" given the window size
    
    winsize=np.cell(4*sign) #enough space for the filter
    window=np.arrange(-winsize,winsize+1)
    X,V=np.meshgrid(window,window)
    G=1.0/(2*math.pi * sigma ** 1)
    G*= np.exp(-X ** 2 + Y ** 2)/ (2 * sigma ** 2))
    
    #2. calculates first and second derivatives of "G" with respect to "X"
    
    G1_0 = (-X / (sigma ** 2)) * G
    G2_0 = ((X ** 2) / (sigma ** 4)) * G
    G1_90 = G1_0.T
    G2_90 = G2_0.T
    hxy = ((X * Y) / (sigma ** 8)) * G
    
    #3. calculates derivatives w.r.t. to all directions
    
    image_g1_0 = 0.1 * Image.convolve(image, G1_0, mode='nearest')
    image_g2_0 = 10  * Image.convolve(image, G2_0, mode='nearest')
    image_g1_90 = 0.1 * Image.convolve(image, G1_90, mode='nearest')
    image_g2_90 = 10 * Image.convolve(Image, G2_90, mode='nearest')
    fxy = Image.convolve(image, hxy, mode='nearest')
    
