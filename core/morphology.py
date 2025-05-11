import cv2
import os
import numpy as np

def apply_morph_operation(image_path, operation, kernel_size=3, iterations=1):
    img = cv2.imread(image_path, 0)  # Grayscale
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    op_map = {
        "erosion": lambda i, k: cv2.erode(i, k, iterations=iterations),
        "dilation": lambda i, k: cv2.dilate(i, k, iterations=iterations),
        "opening": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_OPEN, k),
        "closing": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_CLOSE, k),
        "gradient": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_GRADIENT, k),
        "tophat": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_TOPHAT, k),
        "blackhat": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_BLACKHAT, k),
    }

    if operation not in op_map:
        raise ValueError("Invalid morphological operation.")

    processed = op_map[operation](img, kernel)

    # Convert grayscale to BGR for GUI display compatibility
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{operation}_result.jpg")
    cv2.imwrite(output_path, processed_bgr)

    return output_path

