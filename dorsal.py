# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:17:18 2021

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
from skimage.morphology import convex_hull_image
from skimage.morphology import *
import numpy as np
import csv




match_count=0
match_not_found_count=0
def normalize_pixel(x, v0, v, m, m0):
    dev_coeff = sqrt((v0 * ((x - m) ** 2)) / v)
    return m0 + dev_coeff if x > m else m0 - dev_coeff


def normalize(im, m0, v0):
    m = np.mean(im)
    v = np.std(im) ** 2
    (y, x) = im.shape
    normilize_image = im.copy()
    for i in range(x):
        for j in range(y):
            normilize_image[j, i] = normalize_pixel(im[j, i], v0, v, m, m0)

    return normilize_image


def normalise(img):
    return (img - np.mean(img)) / (np.std(img))


def create_segmented_and_variance_images(im, w, threshold=.2):
    (y, x) = im.shape
    threshold = np.std(im) * threshold

    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # apply threshold
    mask[image_variance < threshold] = 0

    # smooth mask with a open/close morphological filter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w * 2, w * 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # normalize segmented image
    segmented_image *= mask
    im = normalise(im)
    mean_val = np.mean(im[mask == 0])
    std_val = np.std(im[mask == 0])
    norm_img = (im - mean_val) / (std_val)

    return segmented_image, norm_img, mask


def calculate_angles(im, W, smoth=False):
    j1 = lambda x, y: 2 * x * y
    j2 = lambda x, y: x ** 2 - y ** 2
    j3 = lambda x, y: x ** 2 + y ** 2

    (y, x) = im.shape

    sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ySobel = np.array(sobelOperator).astype(np.int)
    xSobel = np.transpose(ySobel).astype(np.int)

    result = [[] for i in range(1, y, W)]

    Gx_ = cv2.filter2D(im / 125, -1, ySobel) * 125
    Gy_ = cv2.filter2D(im / 125, -1, xSobel) * 125

    for j in range(1, y, W):
        for i in range(1, x, W):
            nominator = 0
            denominator = 0
            for l in range(j, min(j + W, y - 1)):
                for k in range(i, min(i + W, x - 1)):
                    Gx = round(Gx_[l, k])  # horizontal gradients at l, k
                    Gy = round(Gy_[l, k])  # vertial gradients at l, k
                    nominator += j1(Gx, Gy)
                    denominator += j2(Gx, Gy)

            # nominator = round(np.sum(Gy_[j:min(j + W, y - 1), i:min(i + W , x - 1)]))
            # denominator = round(np.sum(Gx_[j:min(j + W, y - 1), i:min(i + W , x - 1)]))
            if nominator or denominator:
                angle = (math.pi + math.atan2(nominator, denominator)) / 2
                orientation = np.pi / 2 + math.atan2(nominator, denominator) / 2
                result[int((j - 1) // W)].append(angle)
            else:
                result[int((j - 1) // W)].append(0)

            # segment image
            # focus_img = im[j:min(j + W, y - 1), i:min(i + W , x - 1)]
            # segmentator = -1 if segmentator/W*W < np.max(focus_img)*

    result = np.array(result)

    if smoth:
        result = smooth_angles(result)

    return result


def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))


def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel


def smooth_angles(angles):
    angles = np.array(angles)
    cos_angles = np.cos(angles.copy() * 2)
    sin_angles = np.sin(angles.copy() * 2)

    kernel = np.array(kernel_from_function(5, gauss))

    cos_angles = cv2.filter2D(cos_angles / 125, -1, kernel) * 125
    sin_angles = cv2.filter2D(sin_angles / 125, -1, kernel) * 125
    smooth_angles = np.arctan2(sin_angles, cos_angles) / 2

    return smooth_angles


def get_line_ends(i, j, W, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, int((-W / 2) * tang + j + W / 2))
        end = (i + W, int((W / 2) * tang + j + W / 2))
    else:
        begin = (int(i + W / 2 + W / (2 * tang)), j + W // 2)
        end = (int(i + W / 2 - W / (2 * tang)), j - W // 2)
    return (begin, end)


def visualize_angles(im, mask, angles, W):
    (y, x) = im.shape
    result = cv2.cvtColor(np.zeros(im.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    mask_threshold = (W - 1) ** 2
    for i in range(1, x, W):
        for j in range(1, y, W):
            radian = np.sum(mask[j - 1:j + W, i - 1:i + W])
            if radian > mask_threshold:
                tang = math.tan(angles[(j - 1) // W][(i - 1) // W])
                (begin, end) = get_line_ends(i, j, W, tang)
                cv2.line(result, begin, end, color=150)

    cv2.resize(result, im.shape, result)
    return result


def gabor_filter(im, orient, freq, kx=0.65, ky=0.65):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    return_img = np.zeros((rows, cols))

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    freq_1d = freq.flatten()
    frequency_ind = np.array(np.where(freq_1d > 0))
    non_zero_elems_in_freq = freq_1d[frequency_ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100
    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.
    sigma_x = 1 / unfreq * kx
    sigma_y = 1 / unfreq * ky
    block_size = np.round(3 * np.max([sigma_x, sigma_y]))
    array = np.linspace(-block_size, block_size, (2 * block_size + 1))
    x, y = np.meshgrid(array, array)

    # gabor filter equation
    reffilter = np.exp(-(((np.power(x, 2)) / (sigma_x * sigma_x) + (np.power(y, 2)) / (sigma_y * sigma_y)))) * np.cos(
        2 * np.pi * unfreq[0] * x)
    filt_rows, filt_cols = reffilter.shape
    gabor_filter = np.array(np.zeros((180 // angleInc, filt_rows, filt_cols)))

    # Generate rotated versions of the filter.
    for degree in range(0, 180 // angleInc):
        rot_filt = scipy.ndimage.rotate(reffilter, -(degree * angleInc + 90), reshape=False)
        gabor_filter[degree] = rot_filt

    # Convert orientation matrix values from radians to an index value that corresponds to round(degrees/angleInc)
    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orient / np.pi * 180 / angleInc)
    for i in range(0, rows // 16):
        for j in range(0, cols // 16):
            if (orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if (orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex

    # Find indices of matrix points greater than maxsze from the image boundary
    block_size = int(block_size)
    valid_row, valid_col = np.where(freq > 0)
    finalind = \
        np.where((valid_row > block_size) & (valid_row < rows - block_size) & (valid_col > block_size) & (
                    valid_col < cols - block_size))

    for k in range(0, np.shape(finalind)[1]):
        r = valid_row[finalind[0][k]];
        c = valid_col[finalind[0][k]]
        img_block = im[r - block_size:r + block_size + 1][:, c - block_size:c + block_size + 1]
        return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r // 16][c // 16]) - 1])

    gabor_img = 255 - np.array((return_img < 0) * 255).astype(np.uint8)

    return gabor_img


def frequest(im, orientim, kernel_size, minWaveLength, maxWaveLength):
    rows, cols = np.shape(im)

    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the angle again.
    cosorient = np.cos(2 * orientim)  # np.mean(np.cos(2*orientim))
    sinorient = np.sin(2 * orientim)  # np.mean(np.sin(2*orientim))
    block_orient = math.atan2(sinorient, cosorient) / 2

    # Rotate the image block so that the ridges are vertical
    rotim = scipy.ndimage.rotate(im, block_orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,
                                 mode='nearest')

    # Now crop the image so that the rotated image does not contain any invalid regions.
    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

    # Sum down the columns to get a projection of the grey values down the ridges.
    ridge_sum = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(ridge_sum, kernel_size, structure=np.ones(kernel_size))
    ridge_noise = np.abs(dilation - ridge_sum);
    peak_thresh = 2;
    maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
    maxind = np.where(maxpts)
    _, no_of_peaks = np.shape(maxind)

    # Determine the spatial frequency of the ridges by dividing the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds, the frequency image is set to 0
    if (no_of_peaks < 2):
        freq_block = np.zeros(im.shape)
    else:
        waveLength = (maxind[0][-1] - maxind[0][0]) / (no_of_peaks - 1)
        if waveLength >= minWaveLength and waveLength <= maxWaveLength:
            freq_block = 1 / np.double(waveLength) * np.ones(im.shape)
        else:
            freq_block = np.zeros(im.shape)
    return (freq_block)


def ridge_freq(im, mask, orient, block_size, kernel_size, minWaveLength, maxWaveLength):
    # Function to estimate the fingerprint ridge frequency across a
    # fingerprint image.
    rows, cols = im.shape
    freq = np.zeros((rows, cols))

    for row in range(0, rows - block_size, block_size):
        for col in range(0, cols - block_size, block_size):
            image_block = im[row:row + block_size][:, col:col + block_size]
            angle_block = orient[row // block_size][col // block_size]
            if angle_block:
                freq[row:row + block_size][:, col:col + block_size] = frequest(image_block, angle_block, kernel_size,
                                                                               minWaveLength, maxWaveLength)

    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)
    ind = np.array(ind)
    ind = ind[1, :]
    non_zero_elems_in_freq = freq_1d[0][ind]
    medianfreq = np.median(non_zero_elems_in_freq) * mask

    return medianfreq


def skeletonize(image_input):
    image = np.zeros_like(image_input)
    image[image_input == 0] = 1.0
    output = np.zeros_like(image_input)

    skeleton = skelt(image)

    output[skeleton] = 255
    cv2.bitwise_not(output, output)

    return output


def thinning_morph(image, kernel):
    """
    Thinning image using morphological operations
    :param image: 2d array uint8
    :param kernel: 3x3 2d array unint8
    :return: thin images
    """
    thining_image = np.zeros_like(image)
    img = image.copy()

    while 1:
        erosion = cv2.erode(img, kernel, iterations=1)
        dilatate = cv2.dilate(erosion, kernel, iterations=1)

        subs_img = np.subtract(img, dilatate)
        cv2.bitwise_or(thining_image, subs_img, thining_image)
        img = erosion.copy()

        done = (np.sum(img) == 0)

        if done:
            break

    # shift down and compare one pixel offset
    down = np.zeros_like(thining_image)
    down[1:-1, :] = thining_image[0:-2, ]
    down_mask = np.subtract(down, thining_image)
    down_mask[0:-2, :] = down_mask[1:-1, ]
    cv2.imshow('down', down_mask)

    # shift right and compare one pixel offset
    left = np.zeros_like(thining_image)
    left[:, 1:-1] = thining_image[:, 0:-2]
    left_mask = np.subtract(left, thining_image)
    left_mask[:, 0:-2] = left_mask[:, 1:-1]
    cv2.imshow('left', left_mask)

    # combine left and down mask
    cv2.bitwise_or(down_mask, down_mask, thining_image)
    output = np.zeros_like(thining_image)
    output[thining_image < 250] = 255

    return output


def minutiae_at(pixels, i, j, kernel_size):
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:

        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
                     (0, 1), (1, 1), (1, 0),  # p8    p4
                     (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),  # p1 p2   p3
                     (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),  # p8      p4
                     (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]  # p7 p6   p5

        values = [pixels[i + l][j + k] for k, l in cells]

        # count crossing how many times it goes from 0 to 1
        crossings = 0
        for k in range(0, len(values) - 1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"

    return "none"


def calculate_minutiaes(im, kernel_size=3):
    biniry_image = np.zeros_like(im)
    biniry_image[im < 10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = im.shape
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}

    # iterate each pixel minutia
    for i in range(1, x - kernel_size // 2):
        for j in range(1, y - kernel_size // 2):
            minutiae = minutiae_at(biniry_image, j, i, kernel_size)
            if minutiae != "none":
                cv2.circle(result, (i, j), radius=2, color=colors[minutiae], thickness=2)

    return result


def poincare_index_at(i, j, angles, tolerance):
    cells = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
             (0, 1), (1, 1), (1, 0),  # p8    p4
             (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):

        # calculate the difference
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180

        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"


def calculate_singularities(im, angles, tolerance, W, mask):
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    colors = {"loop": (0, 0, 255), "delta": (0, 128, 255), "whorl": (255, 153, 255)}

    for i in range(3, len(angles) - 2):  # Y
        for j in range(3, len(angles[i]) - 2):  # x
            # mask any singularity outside of the mask
            mask_slice = mask[(i - 2) * W:(i + 3) * W, (j - 2) * W:(j + 3) * W]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (W * 5) ** 2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none":
                    cv2.rectangle(result, ((j + 0) * W, (i + 0) * W), ((j + 1) * W, (i + 1) * W), colors[singularity],
                                  3)

    return result


def removeSpuriousMinutiae(minutiaeList, img, thresh):
    img = img * 0;
    SpuriousMin = [];
    numPoints = len(minutiaeList);
    D = np.zeros((numPoints, numPoints))
    for i in range(1, numPoints):
        for j in range(0, i):
            (X1, Y1) = minutiaeList[i]['centroid']
            (X2, Y2) = minutiaeList[j]['centroid']

            dist = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2);
            D[i][j] = dist
            if (dist < thresh):
                SpuriousMin.append(i)
                SpuriousMin.append(j)

    SpuriousMin = np.unique(SpuriousMin)
    for i in range(0, numPoints):
        if (not i in SpuriousMin):
            (X, Y) = np.int16(minutiaeList[i]['centroid']);
            img[X, Y] = 1;

    img = np.uint8(img);
    return (img)


def getTerminationBifurcation(img, mask):
    img = img == 255;
    (rows, cols) = img.shape;
    minutiaeTerm = np.zeros(img.shape);
    minutiaeBif = np.zeros(img.shape);

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (img[i][j] == 1):
                block = img[i - 1:i + 2, j - 1:j + 2];
                block_val = np.sum(block);
                if (block_val == 2):
                    minutiaeTerm[i, j] = 1;
                elif (block_val == 4):
                    minutiaeBif[i, j] = 1;

    mask = convex_hull_image(mask > 0)
    mask = erosion(mask, square(5))  # Structuing element for mask erosion = square(5)
    minutiaeTerm = np.uint8(mask) * minutiaeTerm
    return (minutiaeTerm, minutiaeBif)


img = input("Enter the test  image")
img1 = cv2.imread(img, 0)

normalized_img = normalize(img1, float(100), float(100))
threshold_img = normalized_img
threshold_im = cv2.threshold(normalized_img, 127, 255, cv2.THRESH_OTSU)
# cv2.imshow('color_threshold', normalized_img); cv2.waitKeyEx()
plt.imshow(normalized_img, cmap=plt.get_cmap('gray'))
plt.title('Normalized Image')
plt.show()


block_size = 16
# ROI and normalisation
(segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

# orientations
angles = calculate_angles(normalized_img, W=block_size, smoth=False)
orientation_img = visualize_angles(segmented_img, mask, angles, W=block_size)

# find the overall frequency of ridges in Wavelet Domain
freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

# create gabor filter and do the actual filtering
# gabor_img = gabor_filter(normim, angles, freq)
g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

filtered_img = cv2.filter2D(img1, cv2.CV_8UC3, g_kernel)



h, w = g_kernel.shape[:2]
gabor_img = cv2.resize(filtered_img, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
# thinning oor skeletonize
thin_image = skeletonize(gabor_img)
plt.imshow(thin_image, cmap=plt.get_cmap('gray'))
plt.title('Thinning')
plt.show()
# minutias
minutias = calculate_minutiaes(thin_image)
plt.title('Minutias')
plt.imshow(minutias, cmap=plt.get_cmap('gray'))
plt.show()
# singularities
singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)
plt.imshow(singularities_img, cmap=plt.get_cmap('gray'))
plt.title('Singularities')
plt.show()
# visualize pipeline stage by stage
output_imgs = [img1, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
for i in range(len(output_imgs)):
    if len(output_imgs[i].shape) == 2:
        output_imgs[i] = cv2.cvtColor(output_imgs[i], cv2.COLOR_GRAY2RGB)
# results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)
# plt.imshow(results, cmap = plt.get_cmap('gray'))
# plt.show()
#
img = np.uint8(img1 > 128);

skel = skimage.morphology.skeletonize(img)
skel = np.uint8(skel) * 255;

mask = img * 255;
(minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);

minutiaeTerm = skimage.measure.label(minutiaeTerm, 8);
RP = skimage.measure.regionprops(minutiaeTerm)
minutiaeTerm = removeSpuriousMinutiae(RP, np.uint8(img), 10);

BifLabel = skimage.measure.label(minutiaeBif, 8);
TermLabel = skimage.measure.label(minutiaeTerm, 8);

minutiaeBif = minutiaeBif * 0;
minutiaeTerm = minutiaeTerm * 0;

(rows, cols) = skel.shape
DispImg = np.zeros((rows, cols, 3), np.uint8)
DispImg[:, :, 0] = skel;
DispImg[:, :, 1] = skel;
DispImg[:, :, 2] = skel;

RP = skimage.measure.regionprops(BifLabel)
for i in RP:
    (row, col) = np.int16(np.round(i['Centroid']))
    minutiaeBif[row, col] = 1;
    (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
    skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0));

RP = skimage.measure.regionprops(TermLabel)
for i in RP:
    (row, col) = np.int16(np.round(i['Centroid']))
    minutiaeTerm[row, col] = 1;
    (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
    skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255));
# print((rr,cc))
# print(DispImg)
plt.title('Bifurcation and End points')
plt.imshow(DispImg, cmap = plt.get_cmap('gray'))
plt.show()

d1 = []
d2 = []
d3 = []
# filename = input("Enter the file name in which images are present = ")
filename = '/content/drive/MyDrive/PROJECTS/dorsal authentication/dorsal'

for img in glob.glob(filename + '/*.*'):
    try:
        var_img = cv2.imread(img, 0)
        var_img1 = var_img


 

        
        source_representation = img1
        #test_representation =input("Enter the test  image")
        test_representation = var_img1
        # a=findCosineDistance(source_representation, test_representation)
        #b=findEuclideanDistance(source_representation, test_representation)
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        # print(euclidean_distance)

        if euclidean_distance == 0.0:
            #if test_list1 == test_list2:
            print("The image is in the database")
            # cv2_imshow(var_img)
            # plt.title('Dataset Image')
            # plt.imshow(var_img)

    
            break
        else:
            pass
            #print("The image is not in the database")

    except Exception as e:
        print(e)
else:
  print("The image is not in the database")