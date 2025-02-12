import numpy as np
import cv2
import os

img = cv2.imread("Dataset/Coins.jpg")
os.makedirs("Output_Task1", exist_ok=True)


# Converting image to grey scale and blurring it
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (17, 17), 0)

# Otsu's Thresholding for segmentation and finding Connected Components
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels = cv2.connectedComponents(binary)

output = np.zeros((*blurred.shape, 3), dtype=np.uint8)
for label in range(1, num_labels):
    output[labels == label] = 255

# Canny edge detector
outline = cv2.Canny(blurred, 30, 150)

# Contours
(cnts, _) = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total number of coins: {len(cnts)}")

cv2.imshow("Original image",img)
cv2.imwrite("Output_Task1/Original image.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Blurred image and grey scaled image
cv2.imshow("Gray scale", grey)
cv2.imwrite("Output_Task1/Gray scale.jpg", grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("blurred", blurred)
cv2.imwrite("Output_Task1/Blurred.jpg", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Edges obtained after canny edge detector
cv2.imshow("The edges", outline)
cv2.imwrite("Output_Task1/The edges.jpg", outline)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Segmented regions
cv2.imshow("Segmented Regions",output)
cv2.imwrite("Output_Task1/Segmented Regions.jpg",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Contours
cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Result", img)
cv2.imwrite("Output_Task1/Result.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()