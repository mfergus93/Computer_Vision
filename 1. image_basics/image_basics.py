# Matt Ferguson
# ECE 5554 Computer Vision Homework 1
import cv2
import numpy as np

# Requirement 1
image_1=cv2.imread(r'C:\Users\Matt\Desktop\Virginia Tech\animals.png')
image_2=cv2.imread(r'C:\Users\Matt\Desktop\Virginia Tech\stonehenge.png')
image_1=image_1.astype(np.float64)
image_2=image_2.astype(np.float64)


# Requirement 2
print(image_1.size)
print(image_2.size)
print(image_1.shape)
print(image_2.shape)

# Requirement 3
image_1_RGB_mean=np.mean(image_1, axis=(0, 1))
image_2_RGB_mean=np.mean(image_2,axis=(0,1))

# Requirement 4
gray_scale_1=np.mean(image_1,2)
gray_scale_2=np.mean(image_2,2)
gray_scale_1_write=gray_scale_1.astype(np.ubyte)
gray_scale_2_write=gray_scale_2.astype(np.ubyte)
cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\gray_scale_1.png', gray_scale_1_write)
cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\gray_scale_2.png', gray_scale_2_write)

# Requirement 5
gray_scale_mean=(gray_scale_1+gray_scale_2)/2
gray_scale_mean=gray_scale_mean.astype(np.ubyte)
cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\gray_scale_mean.png', gray_scale_mean)

# Requirement 6
gray_scale_max=np.maximum(gray_scale_1,gray_scale_2)
gray_scale_max=gray_scale_max.astype(np.ubyte)
cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\gray_scale_max.png', gray_scale_max)

# Requirement 7
gray_scale_absdif=np.abs(np.subtract(gray_scale_1,gray_scale_2))
gray_scale_absdif=gray_scale_absdif.astype(np.ubyte)
cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\gray_scale_absdif.png', gray_scale_absdif)


