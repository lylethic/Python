import numpy as np
import cv2
import matplotlib.pyplot as plt

# uint8	Unsigned integer (0 to 255) So nguyen ko dau tu 0 - 255

# Load the image in BGR format
image = cv2.imread('./image/bae-seok-ryu.png')
print("Size of image: ", image.shape)

pixel = image[100:200, 100] = [0, 255, 0]

print("Pixel: ", pixel)

# Gray image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel | cv2.CV_64F The depth of the output image (64-bit floating point).
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1 ,ksize=5)

# Tính độ lớn của gradient 
edges = np.sqrt(sobel_x**2 + sobel_y**2)

# chuan hoa va convert sang uint8
edges = (edges / edges.max() * 255).astype(np.uint8)

# kernel
kernel = np.ones((3,3))

# Dao nguoc mau anh
invert_image = 255-image

threshold_image = np.where(image > 128, 255, 0).astype(np.uint8)

flip_image = np.flip(image, axis=1)

rotated_image = np.rot90(image)

blue_channel = image[:, :, 0]

swapped_image = image[:, :, [2,1,2]]

blur_image = cv2.filter2D(gray_image, -1, kernel)

cv2.imshow('Image', swapped_image)

# Save the modified image
cv2.imwrite('modified_image.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()