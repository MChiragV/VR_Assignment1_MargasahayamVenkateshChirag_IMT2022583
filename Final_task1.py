# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("Dataset/Coins.jpg")

# Convert the image to grey scale and blur it
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (17, 17), 0)  # Stronger blur effect


# Apply Otsu's Thresholding for segmentation and find the Connected Components
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels = cv2.connectedComponents(binary)

output = np.zeros((*blurred.shape, 3), dtype=np.uint8)

for label in range(1, num_labels):
    output[labels == label] = 255

# Canny edge detector
outline = cv2.Canny(blurred, 30, 150)

# find the contours
(cnts, _) = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total number of coins: {len(cnts)}")


cv2.imshow("Original image",img)
cv2.imwrite("Original image.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Blurred image and grey scaled image
cv2.imshow("Gray scale", grey)
cv2.imwrite("Gray scale.jpg", grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("blurred", blurred)
cv2.imwrite("Blurred.jpg", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edges obtained after canny edge detector
cv2.imshow("The edges", outline)
cv2.imwrite("The edges.jpg", outline)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Segmented regions
cv2.imshow("Segmented Regions",output)
cv2.imwrite("Segmented Regions.jpg",output)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# Contours: -1 will draw all contours
cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Result", img)
cv2.imwrite("Result.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()