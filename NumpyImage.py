import numpy as np
import cv2

image = cv2.imread('./image/bae-seok-ryu.png')
print("Size of image: ", image.shape)

pixel = image[100:200, 100] = [0, 255, 0]

print("Pixel: ", pixel)

image2 = 255-image

threshold_image = np.where(image > 199, 255, 0).astype(np.uint8)

flip_image = np.flip(image, axis=1)

rotated_image = np.rot90(image)

blue_channel = image[:, :, 0]

swapped_image = image[:, :, [2,1,0]]

cv2.imshow('Original and Grayscale', swapped_image)
cv2.imwrite('./image/bae-seok-ryu.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()