import cv2
import matplotlib.pyplot as plt
import numpy as np

## Write a program to draw a rectangle, circle, and a line on an image.

# Read the image
image = cv2.imread('./image/bae-seok-ryu.png')
print(f"[INFOR] image size: {image.shape[:2]}")

# Top-left
start_point = (280,120)
# Botton-Right
end_point = (432,222)

color = (0,0,0)
# Do day
thickless = 12

cv2.rectangle(image, start_point, end_point, color, thickless,199)
cv2.line(image, (50, 50), (250, 250), (0, 255, 0), 5)
cv2.circle(image, (368, 368), 100, (12,155,222), -1)

cv2.imshow('Original and Grayscale', image)

cv2.waitKey(0)
cv2.destroyAllWindows()