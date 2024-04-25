import cv2
import numpy as np

image = cv2.imread('./data/our_dataset/patches/planda.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary mask
_, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

# Display the mask
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the mask
cv2.imwrite('./data/our_dataset/patches/planda_mask.png', mask)