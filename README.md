# Visual Recognition Assignment-1
Name: Margasahayam Venkatesh Chirag <br>
Roll number: IMT2022583

## Question-1: Coin detection and counting
The image "Dataset/Coins.jpg" is considered.
### Explanation along with methods chosen:
### a. Detect all coins in the image
The image has been converted to grayscale and blurred. Then, it is passed through Canny edge detector, and the countours are obtained. Canny edge detector was chosen since it was a good detector for cases like this, where clear edges are visible and where noise is minimal.
For better working, incase the detector detects the engravings on the coin, the Gaussian(used in blurring) can be made stronger.
### b. Segmentation of each coin
The coins have been segmented using Otsu's segmentation method.
Otsu's segmentation is used since it was able to give better regions than other methods.
### c. Count total number of coins
The total number of coins is obtained from number of countours and is printed at the end.
### How to run the code:
```
python3 Final_task1.py
```
### Dependencies:
Ensure the following Python libraries are installed:
``` pip install numpy opencv-python matplotlib ```

## Question-2: Panorama construction from multiple overlapping images
### Explanation along with methods chosen:
### a. Extract key points
The image is converted to grayscale. The keypoints and desciptors have been computed using SIFT method. Then, Brute-Force Matcher and KNN is used to find the 2 best matches for each descriptor. Good matches are retained if the distance(similarity) between the 2 descriptors is less than 75% (lower distance => better match between the keypoints) (Based on Lowe's Ratio test). Coordinates of keypoints that matched successfully are extracted and used to compute homography, which is done using RANSAC. The "mask" variable tells which matches are inliers.
### b. Image stitching
Perspective transformation aligns img1 with img2 using the homography matrix. img2 is placed in the stitched output, thus properly blending.
### How to run the code:
```
python3 Final_task2.py
```
