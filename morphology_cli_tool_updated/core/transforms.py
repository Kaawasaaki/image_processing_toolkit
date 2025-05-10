import cv2
import os
import numpy as np

def apply_transform(image_path, operation, **kwargs):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)

    h, w = img.shape[:2]

    if operation == "rotate":
        angle = kwargs.get('angle', 45)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed = cv2.warpAffine(img, M, (w, h))
    elif operation == "scale":
        fx = kwargs.get('fx', 1.5)
        fy = kwargs.get('fy', 1.5)
        processed = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    elif operation == "translate":
        tx = kwargs.get('tx', 50)
        ty = kwargs.get('ty', 50)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        processed = cv2.warpAffine(img, M, (w, h))
    elif operation == "flip":
        flip_code = kwargs.get('flip_code', 1)  # 0=vertical, 1=horizontal, -1=both
        processed = cv2.flip(img, flip_code)
    elif operation == "affine":
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        processed = cv2.warpAffine(img, M, (w, h))
    elif operation == "perspective":
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        processed = cv2.warpPerspective(img, M, (300, 300))
    elif operation == "crop":
        x = kwargs.get('x', 100)
        y = kwargs.get('y', 100)
        width = kwargs.get('width', 200)
        height = kwargs.get('height', 200)
        processed = img[y:y+height, x:x+width]
    else:
        raise ValueError("Invalid transform operation.")

    # For crop operation, we can't stack images of different sizes
    if operation == "crop":
        combined = processed
    else:
        combined = np.hstack((img, processed))

    output_path = os.path.join(output_dir, f"{operation}_result.jpg")
    cv2.imwrite(output_path, combined)

    return output_path