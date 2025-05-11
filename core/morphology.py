
import cv2
import os
import numpy as np

def apply_transform(image_path, operation, **kwargs):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    h, w = img.shape[:2]
    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)

    processed = None

    if operation == "rotate":
        angle = float(kwargs.get('angle', 45))
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed = cv2.warpAffine(img, M, (w, h))

    elif operation == "scale":
        fx = float(kwargs.get('fx', 1.5))
        fy = float(kwargs.get('fy', 1.5))
        processed = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    elif operation == "translate":
        tx = int(kwargs.get('tx', 50))
        ty = int(kwargs.get('ty', 50))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        processed = cv2.warpAffine(img, M, (w, h))

    elif operation == "flip":
        flip_code = int(kwargs.get('flip_code', 1))
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
        x = int(kwargs.get('x', 100))
        y = int(kwargs.get('y', 100))
        width = int(kwargs.get('width', 200))
        height = int(kwargs.get('height', 200))
        processed = img[y:y+height, x:x+width]

    else:
        raise ValueError(f"Invalid transform operation: {operation}")

    output_path = os.path.join(output_dir, f"{operation}_result.jpg")
    cv2.imwrite(output_path, processed)

    return output_path
