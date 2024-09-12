import cv2
import matplotlib.pyplot as plt
import numpy as np

## Blurring an Image

# Read the image
image = cv2.imread('./image/bae-seok-ryu.png', cv2.IMREAD_UNCHANGED)
print(f"[INFOR] image size: {image.shape[:2]}")

dst = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)

cv2.imshow('Original and Grayscale', np.hstack((image, dst)))

cv2.waitKey(0)
cv2.destroyAllWindows()