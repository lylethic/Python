import cv2
import matplotlib.pyplot as plt
import numpy as np

## Candy Edge detection an Image

# Read the image
image = cv2.imread('./image/bae-seok-ryu.png', cv2.IMREAD_UNCHANGED)
print(f"[INFOR] image size: {image.shape[:2]}")

edge = cv2.Canny(image, 50, 1, 5, L2gradient=True)

cv2.imshow('Original and Grayscale', edge)

cv2.waitKey(0)
cv2.destroyAllWindows()