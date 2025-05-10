import cv2
import os
import numpy as np

def apply_feature_detection(image_path, operation, **kwargs):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)

    if operation == "harris_corners":
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        processed = img.copy()
        processed[dst > 0.01 * dst.max()] = [0, 0, 255]
    elif operation == "shi_tomasi":
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)
        processed = img.copy()
        for i in corners:
            x, y = i.ravel()
            cv2.circle(processed, (x, y), 3, (0, 255, 0), -1)
    elif operation == "sift":
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        processed = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif operation == "surf":
        surf = cv2.xfeatures2d.SURF_create(400)
        kp, des = surf.detectAndCompute(gray, None)
        processed = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    elif operation == "orb":
        orb = cv2.ORB_create()
        kp = orb.detect(gray, None)
        processed = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    elif operation == "blob":
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        processed = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif operation == "contours":
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    else:
        raise ValueError("Invalid feature detection operation.")

    combined = np.hstack((img, processed))
    output_path = os.path.join(output_dir, f"{operation}_result.jpg")
    cv2.imwrite(output_path, combined)

    return output_path