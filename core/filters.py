import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def calculate_metrics(original, processed):
    """Compute image quality metrics"""
    if original.shape != processed.shape:
        return {}  # Skip if dimensions differ
    
    metrics = {}
    
    # MSE (Mean Squared Error)
    metrics['mse'] = np.mean((original - processed) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if metrics['mse'] == 0:
        metrics['psnr'] = float('inf')
    else:
        metrics['psnr'] = cv2.PSNR(original, processed)
    
    # SSIM (Structural Similarity)
    if len(original.shape) == 3:  # Color image
        ssim_val, _ = ssim(original, processed, full=True, multichannel=True)
    else:  # Grayscale
        ssim_val, _ = ssim(original, processed, full=True)
    metrics['ssim'] = ssim_val
    
    return metrics

def plot_comparison(original, processed, metrics, output_path):
    """Generate a comparison plot with metrics"""
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Original vs. Processed
    plt.subplot(1, 2, 1)
    if len(original.shape) == 2:
        plt.imshow(original, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if len(processed.shape) == 2:
        plt.imshow(processed, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image')
    plt.axis('off')
    
    # Add metrics text
    metric_text = "\n".join([f"{k.upper()}: {v:.2f}" for k, v in metrics.items()])
    plt.gcf().text(0.5, 0.02, metric_text, ha='center', fontsize=10)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def apply_filter(image_path, operation, **kwargs):
    img = cv2.imread(image_path) if operation in ['color_hist_equal', 'color_sepia'] else cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize processed image
    processed = img.copy()  # Default initialization, replace with actual processing

    # Apply the requested filter operation
    # Add your filter operations here based on the 'operation' parameter

    # Calculate metrics and generate plots
    if img.shape == processed.shape:
        metrics = calculate_metrics(img, processed)
        plot_path = os.path.join(output_dir, f"{operation}_comparison.png")
        plot_comparison(img, processed, metrics, plot_path)
    
    # Save side-by-side image (as before)
    combined = np.hstack((img, processed)) if img.shape == processed.shape else processed
    output_path = os.path.join(output_dir, f"{operation}_result.jpg")
    cv2.imwrite(output_path, combined)
    
    return output_path, metrics if 'metrics' in locals() else {}