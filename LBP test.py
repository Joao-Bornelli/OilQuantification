import cv2
import numpy as np

def compute_lbp(image, radius=1, neighbors=8):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Initialize the LBP image
    lbp_image = np.zeros_like(gray)

    # Iterate through each pixel in the image
    for y in range(radius, gray.shape[0] - radius):
        for x in range(radius, gray.shape[1] - radius):
            # Get the center pixel value
            center_pixel = gray[y, x]

            # Initialize the binary pattern
            binary_pattern = 0

            # Iterate through the neighbor pixels
            for i in range(neighbors):
                # Calculate the coordinates of the neighbor pixel
                theta = 2 * np.pi * i / neighbors
                x_offset = int(round(radius * np.cos(theta)))
                y_offset = int(round(radius * np.sin(theta)))

                # Get the value of the neighbor pixel
                neighbor_pixel = gray[y + y_offset, x + x_offset]

                # Update the binary pattern
                binary_pattern |= (neighbor_pixel >= center_pixel) << i

            # Convert the binary pattern to a decimal value and assign it to the LBP image
            lbp_image[y, x] = binary_pattern

    return lbp_image

# Load an image
image = cv2.imread(r'C:\Code\OilVisualization\Images\usar\SemOleo (2).jpg')

# Compute the LBP image
lbp_image = compute_lbp(image)

# Display the original and LBP images
cv2.imshow('Original Image', image)
cv2.imshow('LBP Image', lbp_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()