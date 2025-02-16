import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("Output_Task2", exist_ok=True)

def stitch_images(img1, img2, a, b):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(gray1, None)
    key_points2, descriptor2 = sift.detectAndCompute(gray2, None)
    draw_keypoints(img1, key_points1, f"Keypoints in {a}")
    draw_keypoints(img2, key_points2, f"Keypoints in {b}")
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2) #Top 2 best matches for each descriptor
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance] # 0.75 comes from Lowe's ratio test
    src_pts = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) #Reshape to use it in findHomography built-in function
    dst_pts = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) #Reshape to use it in findHomography built-in function
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    draw_matches(img1, img2, key_points1, key_points2, good_matches, "All Matches (Inliers + Outliers)")
    draw_matches(img1, img2, key_points1, key_points2, inlier_matches, "Inlier Matches Only")
    
    rows_img1, cols_img1 = img1.shape[:2]
    rows_img2, cols_img2 = img2.shape[:2]
    pts_img1 = np.float32([[0, 0], [0, rows_img1], [cols_img1, rows_img1], [cols_img1, 0]]).reshape(-1, 1, 2)
    pts_img2 = np.float32([[0, 0], [0, rows_img2], [cols_img2, rows_img2], [cols_img2, 0]]).reshape(-1, 1, 2)
    pts_img1_trans = cv2.perspectiveTransform(pts_img1, H) #Used for warping
    all_pts = np.concatenate((pts_img1_trans, pts_img2), axis=0)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    translation = np.array([[1, 0, -x_min],[0, 1, -y_min],[0, 0, 1]])
    H_adjusted = translation.dot(H) #Making coordinates positive(i.e., visible range) by adjusting H
    
    warp1 = cv2.warpPerspective(img1, H_adjusted, (canvas_width, canvas_height)) #Warp img1 onto the canvas using adjusted H matrix
    mask1 = cv2.warpPerspective(np.ones((rows_img1, cols_img1), dtype=np.uint8) * 255,H_adjusted, (canvas_width, canvas_height)) #Create mask for blending process
    canvas2 = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    mask2 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    x_offset = -x_min
    y_offset = -y_min
    canvas2[y_offset:y_offset+rows_img2, x_offset:x_offset+cols_img2] = img2 #Place img2 correctly onto the canvas.
    mask2[y_offset:y_offset+rows_img2, x_offset:x_offset+cols_img2] = 255 #Create mask for blending process(white for img2 region)
    
    mask1_bin = (mask1 != 0).astype(np.uint8) #Non-zero value of binary mask means 1
    mask2_bin = (mask2 != 0).astype(np.uint8) #Non-zero value of binary mask means 1
    dist1 = cv2.distanceTransform(mask1_bin, cv2.DIST_L2, 5) #Distance transform for 5X5 mask 
    dist2 = cv2.distanceTransform(mask2_bin, cv2.DIST_L2, 5) #Distance transform for 5X5 mask
    final_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    
    overlap = (mask1_bin == 1) & (mask2_bin == 1) #Both img1 and img2
    only1   = (mask1_bin == 1) & (mask2_bin == 0) #Only img1
    only2   = (mask2_bin == 1) & (mask1_bin == 0) #Only img2
    denom = dist1 + dist2
    denom[denom == 0] = 1 #Avoid divide-by-zero
    w1 = dist1 / denom #As distance from edge increases, weight should increase.
    w2 = dist2 / denom #As distance from edge increases, weight should increase.
    final_img[overlap] = warp1[overlap] * w1[overlap, None] + canvas2[overlap] * w2[overlap, None] #Warping
    final_img[only1]   = warp1[only1]
    final_img[only2]   = canvas2[only2]
    final_img = np.clip(final_img, 0, 255).astype(np.uint8) #Fixing pixels with extreme values
    
    union_mask = cv2.bitwise_or(mask1, mask2)#If either of the masks have value 1, then on fixing also, that pixel should be value 1.
    pts = cv2.findNonZero(union_mask)
    if pts is None:
        return final_img #Image has no zero-pixels i.e., perfect already(rarely happens)
    pts = pts.reshape(-1, 2)#(x,y) coords
    
    # Compute extreme corners based on coordinate sums/differences.
    s = pts.sum(axis=1)
    top_left = pts[np.argmin(s)] #Since it is top left since x-coord is low, y-coord is low(Note that we are accessing from top right), sum is least
    bottom_right = pts[np.argmax(s)] #Since it is bottom right since x-coord is high, y-coord is high(Note that we are accessing from top right), sum is most
    diff = np.diff(pts, axis=1).reshape(-1)
    top_right = pts[np.argmin(diff)] #For same reasons stated above
    bottom_left = pts[np.argmax(diff)] #For same reasons stated above
    quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32") #Quadrilateral enclosing the stitched content.
    
    maxWidth = int(max(np.linalg.norm(bottom_right - bottom_left),np.linalg.norm(top_right - top_left)))#Max width(from top and bottom edge lengths)
    maxHeight = int(max(np.linalg.norm(top_right - bottom_right),np.linalg.norm(top_left - bottom_left)))#Max height(from right and left edge lengths)
    dst_quad = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")#Expected canvas size
    M_rect = cv2.getPerspectiveTransform(quad, dst_quad)#Make it into expected dimensions
    final_image = cv2.warpPerspective(final_img, M_rect, (maxWidth, maxHeight))
    return final_image

def draw_keypoints(image, keypoints, title):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def draw_matches(img1, img2, key_points1, key_points2, matches, title):
    match_img = cv2.drawMatches(img1, key_points1, img2, key_points2, matches, None, matchColor=(0, 255, 0))
    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

part1 = cv2.imread('Dataset/PartA.jpg')
part2 = cv2.imread('Dataset/PartB.jpg')
stitched = stitch_images(part1, part2, "PartA", "PartB")
cv2.imwrite('Output_Task2/Stitched_image.jpg', stitched)
