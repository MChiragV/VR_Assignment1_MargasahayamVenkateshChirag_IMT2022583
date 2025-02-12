# Visual Recognition Assignment-1
Name: Margasahayam Venkatesh Chirag <br>
Roll number: IMT2022583

## Question-1: Coin detection and counting
The image "Dataset/Coins.jpg" is considered.
### a. Detect all coins in the image
The image has been converted to grayscale and blurred. Then, it is passed through Canny edge detector, and the countours are obtained.
### b. Segmentation of each coin
The coins have been segmented using Otsu's segmentation method.
### c. Count total number of coins
The total number of coins is obtained from number of countours and is printed at the end.

### How to run the code:
```
python3 Final_task1.py
```
### Methods chosen:
#### For coin detection:
Canny edge detector was chosen since it was a good detector for cases like this, where clear edges are visible and where noise is minimal.
For better working, incase the detector detects the engravings on the coin, the Gaussian(used in blurring) can be made stronger.
#### For segmentation:
Otsu's segmentation is used since it was able to give better regions than other methods.
## Question-2: Panorama construction from multiple overlapping images
### a. Extract key points
The image is converted to grayscale. The keypoints and desciptors have been computed using SIFT method. Then, Brute-Force Matcher and KNN is used to find the 2 best matches for each descriptor. <b>Good matches</b> are retained if the distance(similarity) between the 2 descriptors is less than 75% (lower distance => better match between the keypoints) (Based on Lowe's Ratio test). 
### b. Image stitching

### How to run the code:
```
python3 Final_task2.py
```
