import os
import cv2
import numpy as np
# Assuming your core modules are structured as shown
from core.morphology import apply_morph_operation
from core.filters import apply_filter
from core.transforms import apply_transform # Ensure this module exists and function is defined
from core.features import apply_feature_detection
try:
    from core.invisibility import run_invisibility_menu
except ImportError:
    print("Warning: core.invisibility module or run_invisibility_menu not found. Invisibility Cloak feature will be disabled in CLI.")
    def run_invisibility_menu(): # Placeholder
        print("Invisibility Cloak feature is not available.")

import argparse
# This import is correct as main.py is in the root, and gui_launcher.py is in gui/
from gui.gui_launcher import main as launch_gui


def generate_sample_images():
    # (generate_sample_images function as provided in your main.py)
    # ... (omitted for brevity, use your existing function) ...
    os.makedirs("images", exist_ok=True)

    # 1. Morphology (text + noise)
    text_noise = np.zeros((100, 300), dtype=np.uint8)
    cv2.putText(text_noise, 'TEST', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 5)
    noise = (np.random.rand(100, 300) < 0.05).astype(np.uint8) * 255
    text_noise = cv2.bitwise_or(text_noise, noise)
    cv2.imwrite("images/text_noise.jpg", text_noise)

    # 2. Natural photo-like image (gradient + shapes)
    photo = np.zeros((100, 300, 3), dtype=np.uint8)
    for i in range(photo.shape[0]):
        photo[i] = (i * 255 // photo.shape[0], 128, 255 - i * 255 // photo.shape[0])
    cv2.circle(photo, (150, 50), 30, (255, 255, 255), -1)
    cv2.imwrite("images/photo.jpg", photo)

    # 3. Shapes (for contour detection)
    shapes = np.zeros((150, 300), dtype=np.uint8)
    cv2.rectangle(shapes, (20, 20), (100, 100), 255, -1)
    cv2.circle(shapes, (200, 75), 35, 255, -1)
    cv2.line(shapes, (250, 30), (290, 120), 255, 3)
    cv2.imwrite("images/shapes.png", shapes)

    # 4. Portrait-style (synthetic grayscale with detail)
    portrait = np.tile(np.linspace(60, 200, 300).astype(np.uint8), (100, 1))
    # portrait = cv2.equalizeHist(portrait) # This might make it too contrasty for some filters
    cv2.imwrite("images/portrait.jpg", portrait)

    # 5. Feature detection sample (chessboard pattern)
    chessboard = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 25):
        for j in range(0, 200, 25):
            if (i//25 + j//25) % 2 == 0:
                chessboard[i:i+25, j:j+25] = 255
    cv2.imwrite("images/chessboard.png", chessboard)
    print("Sample images generated in 'images/' directory.")


def select_sample_image(category):
    # (select_sample_image function as provided in your main.py)
    # ... (omitted for brevity, use your existing function) ...
    image_paths = {
        "morphology": "images/text_noise.jpg",
        "filters": "images/photo.jpg",
        "transform": "images/shapes.png", 
        "features": "images/chessboard.png", 
        "contrast": "images/portrait.jpg", # Added for CLI if needed
    }
    default_image = image_paths.get(category, "images/photo.jpg") # Default to photo.jpg

    print("\nImage Options:")
    print(f"1. Use default sample for '{category}': {os.path.basename(default_image)}")
    print("2. Enter custom image path")
    choice = input("Select (1-2): ")

    if choice == "1":
        if not os.path.exists(default_image):
            print(f"Warning: Default image '{default_image}' not found. Please generate samples or check path.")
            return None
        return default_image
    else:
        custom_path = input("Enter full path to your image: ")
        if not os.path.exists(custom_path):
            print(f"Warning: Custom path '{custom_path}' does not exist.")
            return None
        return custom_path


def run_morphology_menu():
    # (run_morphology_menu function as provided in your main.py)
    # ... (omitted for brevity, use your existing function and ensure core functions are robust) ...
    print("\n-- Morphological Operations --")
    options = {"1": "erosion", "2": "dilation", "3": "opening", "4": "closing", "5": "gradient", "6": "tophat", "7": "blackhat"}
    for k, v in options.items(): print(f"{k}. {v.capitalize()}")
    choice = input(f"Select operation (1-{len(options)}): ")
    if choice in options:
        image_path = select_sample_image("morphology")
        if not image_path: return
        try:
            kernel_size = int(input("Enter kernel size (odd, default 3): ") or 3)
            iterations = int(input("Enter iterations (default 1): ") or 1)
            out_path = apply_morph_operation(image_path, options[choice], kernel_size=kernel_size, iterations=iterations)
            print(f"✅ Done. Saved to: {out_path}")
        except Exception as e: print(f"Error: {e}")
    else: print("Invalid choice.")


def run_filters_menu():
    # (run_filters_menu function as provided in your main.py)
    # ... (omitted for brevity, use your existing function and ensure core functions are robust) ...
    print("\n-- Filter Operations --")
    options = {
        "1": "blur", "2": "gaussian", "3": "median", "4": "bilateral", "5": "sobel", 
        "6": "laplacian", "7": "canny", "8": "threshold", "9": "adaptive_threshold", 
        "10": "equalize", "11": "color_hist_equal", "12": "color_sepia"
    } # Expanded to match more from GUI options
    for k, v in options.items(): print(f"{k}. {v.replace('_', ' ').title()}")
    choice = input(f"Select filter (1-{len(options)}): ")
    if choice in options:
        image_path = select_sample_image("filters")
        if not image_path: return
        try:
            params = {}
            op_name = options[choice]
            if op_name in ["blur", "gaussian", "median", "bilateral", "sobel", "laplacian"]:
                params['kernel_size'] = int(input(f"Enter kernel size for {op_name} (odd, default 5): ") or 5)
            if op_name == "canny":
                params['threshold1'] = int(input("Enter Canny threshold1 (default 100): ") or 100)
                params['threshold2'] = int(input("Enter Canny threshold2 (default 200): ") or 200)
            if op_name == "threshold":
                 params['threshold_value'] = int(input("Enter threshold value (0-255, default 127): ") or 127)
            # Add more param inputs as needed for other filters
            out_path, _ = apply_filter(image_path, op_name, **params)
            print(f"✅ Done. Saved to: {out_path}")
        except Exception as e: print(f"Error: {e}")
    else: print("Invalid choice.")


def run_transform_menu():
    # (run_transform_menu function as provided in your main.py)
    # ... (omitted for brevity, use your existing function and ensure core functions are robust) ...
    print("\n-- Geometric Transformations --")
    options = {"1": "rotate", "2": "scale", "3": "translate", "4": "flip", "5": "affine", "6": "perspective", "7": "crop"}
    for k, v in options.items(): print(f"{k}. {v.capitalize()}")
    choice = input(f"Select transform (1-{len(options)}): ")
    if choice in options:
        image_path = select_sample_image("transform")
        if not image_path: return
        try:
            op_name = options[choice]
            params = {}
            if op_name == "rotate": params['angle'] = float(input("Angle (default 45): ") or 45)
            elif op_name == "scale":
                params['fx'] = float(input("X scale (default 1.5): ") or 1.5)
                params['fy'] = float(input("Y scale (default 1.5): ") or 1.5)
            # Add other params for translate, flip, crop, affine, perspective
            out_path = apply_transform(image_path, op_name, **params)
            print(f"✅ Done. Saved to: {out_path}")
        except Exception as e: print(f"Error: {e}")
    else: print("Invalid choice.")


def run_features_menu():
    # (run_features_menu function as provided in your main.py)
    # ... (omitted for brevity, use your existing function and ensure core functions are robust) ...
    print("\n-- Feature Detection --")
    options = {"1": "harris_corners", "2": "shi_tomasi", "3": "sift", "4": "surf", "5": "orb", "6": "blob", "7": "contours"}
    for k, v in options.items(): print(f"{k}. {v.replace('_', ' ').title()}")
    choice = input(f"Select feature detection (1-{len(options)}): ")
    if choice in options:
        image_path = select_sample_image("features")
        if not image_path: return
        try:
            params = {}
            if options[choice] == "harris_corners":
                 params['threshold'] = float(input("Harris threshold (e.g., 0.01): ") or 0.01)
            # Add params for other feature detectors if needed
            out_path = apply_feature_detection(image_path, options[choice], **params)
            print(f"✅ Done. Saved to: {out_path}")
        except Exception as e: print(f"Error: {e}")
    else: print("Invalid choice.")


def cli_main():
    print("=== Advanced Image Processing CLI Toolkit ===")
    if not os.path.exists("images/photo.jpg"): # Quick check if samples might be missing
        print("Sample images might be missing. Generating them now...")
        generate_sample_images()
    
    while True:
        print("\nMain Menu")
        print("0. Generate Sample Images")
        print("1. Morphological Operations")
        print("2. Filter Operations")
        print("3. Geometric Transformations")
        print("4. Feature Detection")
        print("5. Invisibility Cloak")
        print("6. Exit")
        main_choice = input("Choose (0-6): ")

        if main_choice == "0": generate_sample_images()
        elif main_choice == "1": run_morphology_menu()
        elif main_choice == "2": run_filters_menu()
        elif main_choice == "3": run_transform_menu()
        elif main_choice == "4": run_features_menu()
        elif main_choice == "5": run_invisibility_menu()
        elif main_choice == "6": print("Goodbye!"); break
        else: print("Invalid option.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Toolkit")
    parser.add_argument("--mode", choices=["cli", "gui"], default="cli", 
                        help="Choose execution mode: Command Line Interface (cli) or Graphical User Interface (gui)")
    args = parser.parse_args()

    if args.mode == "gui":
        print("Launching GUI mode...")
        launch_gui()
    else:
        cli_main()
