import cv2
import os
import numpy as np

def apply_feature_detection(image_path, operation, **kwargs):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()

    if operation == "harris_corners":
        threshold = kwargs.get("threshold", 0.01)
        gray_f = np.float32(gray)
        dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        output[dst > threshold * dst.max()] = [0, 0, 255]

    elif operation == "shi_tomasi":
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners) if corners is not None else []
        for i in corners:
            x, y = i.ravel()
            cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

    elif operation == "sift":
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        output = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    elif operation == "surf":
        try:
            surf = cv2.xfeatures2d.SURF_create(400)
            kp, _ = surf.detectAndCompute(gray, None)
            output = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        except AttributeError:
            raise RuntimeError("SURF not available. Install opencv-contrib-python.")

    elif operation == "orb":
        orb = cv2.ORB_create()
        kp = orb.detect(gray, None)
        output = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    elif operation == "blob":
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        output = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    elif operation == "contours":
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    else:
        raise ValueError("Invalid feature detection operation.")

    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{operation}_result.jpg")
    cv2.imwrite(output_path, output)

    return output_path
