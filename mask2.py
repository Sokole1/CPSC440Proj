import cv2
import numpy as np

# Create a black image (all zeros) with the same dimensions as the original image
image = cv2.imread('./data/our_dataset/patches/planda.png')
mask = np.zeros_like(image[:,:,0])

# Create a window and callback function for mouse events
window_name = 'Draw Mask'
cv2.namedWindow(window_name)

# Mouse callback function
def draw_mask(event, x, y, flags, param):
    global mask
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(mask, (x, y), 10, (255), -1)

# Set mouse callback function for the window
cv2.setMouseCallback(window_name, draw_mask)

while True:
    # Display the mask
    cv2.imshow(window_name, mask)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Exit if 'q' is pressed
    if key == ord('q'):
        break

# Save the mask
cv2.imwrite('./data/our_dataset/patches/planda_mask.png', mask)

# Close all windows
cv2.destroyAllWindows()
