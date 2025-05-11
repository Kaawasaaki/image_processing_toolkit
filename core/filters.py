import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def calculate_metrics(original, processed):
    if original.shape != processed.shape:
        return {}
    metrics = {}
    metrics['mse'] = np.mean((original - processed) ** 2)
    metrics['psnr'] = cv2.PSNR(original, processed) if metrics['mse'] > 0 else float('inf')
    ssim_val, _ = ssim(original, processed, full=True, channel_axis=-1 if len(original.shape) == 3 else None)
    metrics['ssim'] = ssim_val
    return metrics

def plot_comparison(original, processed, metrics, output_path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title('Processed')
    plt.axis('off')

    text = "\n".join([f"{k.upper()}: {v:.2f}" for k, v in metrics.items()])
    plt.gcf().text(0.5, 0.02, text, ha='center', fontsize=10)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def apply_filter(image_path, operation, **kwargs):
    gray_ops = ["blur", "gaussian", "median", "bilateral", "sobel", "laplacian", "canny", "threshold", "adaptive_threshold", "equalize"]
    color_ops = ["color_hist_equal", "color_sepia"]

    if operation in color_ops:
        img = cv2.imread(image_path)  # color
    else:
        img = cv2.imread(image_path, 0)  # grayscale

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    processed = None
    kernel = kwargs.get("kernel_size", 5)

    # Apply filters
    if operation == "blur":
        processed = cv2.blur(img, (kernel, kernel))
    elif operation == "gaussian":
        processed = cv2.GaussianBlur(img, (kernel, kernel), 0)
    elif operation == "median":
        processed = cv2.medianBlur(img, kernel)
    elif operation == "bilateral":
        processed = cv2.bilateralFilter(img, kernel, 75, 75)
    elif operation == "sobel":
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
        processed = cv2.convertScaleAbs(sobelx + sobely)
    elif operation == "laplacian":
        lap = cv2.Laplacian(img, cv2.CV_64F)
        processed = cv2.convertScaleAbs(lap)
    elif operation == "canny":
        t1 = kwargs.get("threshold1", 100)
        t2 = kwargs.get("threshold2", 200)
        processed = cv2.Canny(img, t1, t2)
    elif operation == "threshold":
        t = kwargs.get("threshold_value", 127)
        _, processed = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    elif operation == "adaptive_threshold":
        processed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    elif operation == "equalize":
        processed = cv2.equalizeHist(img)
    elif operation == "color_hist_equal":
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    elif operation == "color_sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        processed = cv2.transform(img, sepia_filter)
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown filter operation: {operation}")

    # Convert grayscale processed image to BGR for consistent GUI display
    if processed.ndim == 2:
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed_bgr = processed

    # Likewise, original needs to be BGR
    if img.ndim == 2:
        original_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = img

    metrics = calculate_metrics(original_bgr, processed_bgr)
    comparison_path = os.path.join(output_dir, f"{operation}_comparison.png")
    plot_comparison(original_bgr, processed_bgr, metrics, comparison_path)

    result_path = os.path.join(output_dir, f"{operation}_result.jpg")
    cv2.imwrite(result_path, processed_bgr)

    return result_path, metrics
