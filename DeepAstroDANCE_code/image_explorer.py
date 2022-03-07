#import pandas as pd 
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np

import os 

#saving images and labels

y1_image_set = np.load("images/images_Y1_test_150.npy")
y10_image_set = np.load("images/images_Y10_test_150.npy")
labels = np.load("images/labels_test_150.npy")

y1_sample_img = y1_image_set[58]
y10_sample_img = y10_image_set[58]
sample_labels = labels[58] 

img_path = "images/img_plots/"
#print the y1 example images
for i in range(0,len(y1_sample_img)):
    plt.imshow(y1_sample_img[i])
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    plt.savefig("images/img_plots/y1_fig_{num}".format(num=i))

#print the y10 example images
for i in range(0,len(y10_sample_img)):
    plt.imshow(y10_sample_img[i])
    plt.savefig("images/img_plots/y10_fig_{num}".format(num=i))

#printing an example of the label input
print(sample_labels)  # --> output was '[1 0 0]

#printing an example of how the image input looks before making a plot
print(y1_sample_img)
