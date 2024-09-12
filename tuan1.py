import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
image = cv2.imread('./image/bae-seok-ryu.png')

# Lay kich thuoc anh goc
original_height, original_width = image.shape[:2]
print(f"[INFOR] image size: {image.shape[:2]}")

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image
half_img = cv2.resize(image, (original_width // 2, original_height // 2))
double_img = cv2.resize(image, (original_width * 2, original_height * 2))

half = cv2.resize(half_img, (original_width, original_height))
double = cv2.resize(double_img, (original_width, original_height))

# Display using matplotlib

# images = [image, gray, half_img, double_img]
# titles = ['Ảnh gốc', 'Ảnh xám hóa', 'Ảnh một nửa ảnh gốc', 'Ảnh gấp đôi ảnh gốc']
# fig, axs = plt.subplots(1, 4, figsize =(20, 5))

# for ax, img, title in zip(axs, images, titles):
  
#     if len(img.shape) == 2:
#         ax.imshow(img, cmap='gray')
#     else:
#         ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
#     # Set title and remove axis
#     ax.set_title(title)
#     ax.axis('off')
    
# plt.show()


# Display
cv2.imshow('Original and Grayscale', image)
# cv2.imshow('Grayscale', gray)
# cv2.imshow('Half', half_img)
# cv2.imshow('Double', double_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
