# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:40:10 2021

@author: USER
"""

det gabor_tilter(im , orient, treq, kx=0.65, ky=0.65): anglelnc = 3
im = np .double(im) ro•1s, cols	im .shape
return_img = np .zeros((ro•1s, cols))

# Round the array of frequencies to the nearest 0.01 to reduce the
# number of distinct frequencies we have to deal with . freq_ld = freq .flatten ()
frequency_ind = np .array(np .•1here(freq_ld > 0}) non_zero_elems_in_freq = freq_ld[frequency_ind]
non_zero_elems_in_freq = np .double(np .round((non_zero_elems_in_freq •100)))I 100 unfreq = np .unique(non_zero_elems_in_freq )

# Generate filters correspond ing to these distinct frequencies and
# orientations in 'anglelnc' increments. sigma_x = 1 I unfreq •kx
sigma_y = 1 I unfreq •ky
block_size = np .round (3 •np .max ([sigma_x, sigma_y ]))
array = np .linspace(-block_size, block_size, (2 •block_size + 1) )
x, y = np .meshgrid (array, array)

# gabor filter equation
reffilter = np .exp(-(((np .po•1er(x, 2})I (sigma_x * sigma_x)+ (np .potier(y, 2))I (sigma_y * sigma_y ))))* np .cos( 2 * np .pi •unfreq (0]* x)

                                                                                                                     

def getTerminationBifurcation (img, mask): img =  img ==  255;
(rotJS, cols)= img.shape; minutiaeTerm = np .zeros(img .shape); minutiaeBif =  np.zeros(img.shape);

for i in range(1, rows - 1):
for j in range(1, cols - 1}: if (img[i][j]==    1):
block =  img[i - 1:i + 2, j - 1:j + 2]; block_val =  np.sum(block );
if (block_val == 2}:
minutiaeTerm[i, j]=  1; elif (block_val == 4):
minutiaeBif[i, j]= 1;

mask =  convex_hull_image(mask > 0}
mask = erosion(mask, square(5)} # Structuing element for mask erosion	square(5) minutiaeTerm = np .uint8(mask)* minutiaeTerm
return (minutiaeTerm , minutiaeBif)


img = input("Enter the test	image") img1 = cv2.imread(img, 0}

normalized_img = normalize(img1, float(100), float(100})
                           

h, N = g_kernel.shape(:2]
gabor_img = cv2.resize(filtered_img, (3*N , 3*h), interpolation=cv2.INTER_CUBIC)
# thinning oor skeletonize
thin_image = skeletonize(gabor_img)
plt.imsho•1(thin_image, cmap=plt.get_cmap('gray')) plt.title('Thinning ')
plt.shoti()
# minutias
minutias = calculate_minutiaes(thin_image) plt.title('Minutias')
plt.imsho•1(minutias, cmap=plt .get_cmap(•gray ')) plt.shoti()
# singularities
singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask) plt.imsho•1(singularities_img, cmap=plt .get_cmap(•gray'))
plt.title('Singularities') plt.shoti()
# visualize pipeline stage by stage
output_imgs = [img1, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img] for i in range(len(output_imgs)):
if len(output_imgs(i].shape)== 2:
output imgsf il = cv2.cvtColor(output imgsf il, cv2.COLOR GRAY2RGB)
                                                                                                                     