import cv2
import numpy as np
import matplotlib.pyplot as plt

# Just has 0 255 
image  = cv2.imread('./image/alphabet.jpg', 0)
image2  = cv2.imread('./image/ahm.jpeg', 0)


# Erosion (Xoi mon)

# binarize the image 
binarize = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
binarize2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# define the kernel 
kernel = np.ones((5,5), np.uint8)
kernel2 = np.ones((3,3), np.uint8)

# invert the image
invert = cv2.bitwise_not(binarize)

# erode the image | iterations=>(lan lap lai)
erosion = cv2.erode(invert, kernel, iterations=1)

# #Dilation 
dilation = cv2.dilate(invert, kernel2, iterations=1) 

# #Opening: loai bo su nhieu trong anh (La su ket hop giua erosion && dilation)
opening = cv2.morphologyEx(binarize2, cv2.MORPH_OPEN, kernel2, iterations=1) 

# #Closing: loai bo su nhieu trong anh (La su ket hop giua erosion && dilation)
closing = cv2.morphologyEx(binarize2, cv2.MORPH_CLOSE, kernel2, iterations=1) 

cv2.imshow("erosion", erosion)
cv2.waitKey(0)

cv2.imshow("dilation", dilation)
cv2.waitKey(0)

cv2.imshow("opening", opening)
cv2.waitKey(0)

cv2.imshow("closing", closing)
cv2.waitKey(0)

cv2.destroyAllWindows()