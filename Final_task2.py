import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("Output_Task2", exist_ok=True)

def stitch_images(img1, img2, a, b):
    # Grayscale the images
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Keypoints and descriptors are detected using SIFT
    sift = cv2.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(gray1, None)
    key_points2, descriptor2 = sift.detectAndCompute(gray2, None)
    
    draw_keypoints(img1, key_points1, f"Keypoints in {a}")
    draw_keypoints(img2, key_points2, f"Keypoints in {b}")
    
    # Brute-Force Matcher with KNN for best 2 feature descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    
    # Lowe's ratio test to filter good matches among the ones obtained above
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    # Matched keypoint coordinates
    src_pts = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Homography using RANSAC
    final_H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    # Inlier matches based on RANSAC mask
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    
    draw_matches(img1, img2, key_points1, key_points2, good_matches, "All Matches (Inliers + Outliers)")
    draw_matches(img1, img2, key_points1, key_points2, inlier_matches, "Inlier Matches Only")
    
    # Warp the first image and align it with second image
    rows1, cols1 = img2.shape[:2]
    rows2, cols2 = img1.shape[:2]
    
    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points2, final_H)
    
    list_of_points = np.concatenate((points1, transformed_points), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    # Adjust transformation matrix
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(final_H)
    output_img = cv2.warpPerspective(img1, H_translation, (x_max - x_min, y_max - y_min))
    output_img[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = img2
    
    return output_img

def draw_keypoints(image, keypoints, title):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def draw_matches(img1, img2, key_points1, key_points2, matches, title):
    match_img = cv2.drawMatches(img1, key_points1, img2, key_points2, matches, None, matchColor=(0, 255, 0), singlePointColor=None)
    plt.figure()
    plt.imshow(match_img)
    plt.title(title)
    plt.axis("off")
    plt.show()

part1 = cv2.imread('Dataset/Part1.jpg')
part2 = cv2.imread('Dataset/Part2.jpg')
part3 = cv2.imread('Dataset/Part3.jpg')

# Stitch part1 and part2
stitched_1_2 = stitch_images(part1, part2, "Part-1", "Part-2")
# cv2.imwrite('Output_Task2/Image1_2_combined.jpg', stitched_1_2) # You can uncomment this if you want to see the intermediate result.

# Stitch the result with part3, visualize keypoints and matches
final_result = stitch_images(stitched_1_2, part3, "Part-1,2","Part-3")
cv2.imwrite('Output_Task2/Stitched_image.jpg', final_result)