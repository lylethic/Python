import cv2 as cv
import numpy as np

image = cv.imread('./image/ahm.jpeg');

height, width = image.shape[:2]

center = (width/2, height/2)

rotate = cv.getRotationMatrix2D(center=center, angle=45, scale=1)

rotated_img = cv.warpAffine(src=image, M=rotate, dsize=(width, height))

cv.imshow("Rotate image: ", rotated_img)
cv.imwrite("rotate_image.jpg", rotated_img)

cv.waitKey(0)
cv.destroyAllWindows()