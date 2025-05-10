import os
import cv2
import numpy as np
from core.morphology import apply_morph_operation
from core.filters import apply_filter
from core.transforms import apply_transform
from core.features import apply_feature_detection

def generate_sample_images():
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
    portrait = cv2.equalizeHist(portrait)
    cv2.imwrite("images/portrait.jpg", portrait)

    # 5. Feature detection sample (chessboard pattern)
    chessboard = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 25):
        for j in range(0, 200, 25):
            if (i//25 + j//25) % 2 == 0:
                chessboard[i:i+25, j:j+25] = 255
    cv2.imwrite("images/chessboard.png", chessboard)

def select_sample_image(category):
    image_paths = {
        "morphology": "images/text_noise.jpg",
        "filters": "images/photo.jpg",
        "transform": "images/shapes.png",
        "contrast": "images/portrait.jpg",
        "features": "images/chessboard.png"
    }

    print("\nImage Options:")
    print("1. Use default sample image")
    print("2. Enter custom image path")
    choice = input("Select (1-2): ")

    if choice == "1":
        return image_paths.get(category, "images/photo.jpg")
    else:
        return input("Enter full path to your image: ")

def run_morphology_menu():
    print("\n-- Morphological Operations --")
    options = {
        "1": "erosion",
        "2": "dilation",
        "3": "opening",
        "4": "closing",
        "5": "gradient",
        "6": "tophat",
        "7": "blackhat"
    }
    for k, v in options.items():
        print(f"{k}. {v.capitalize()}")
    
    choice = input("Select operation (1-7): ")
    image_path = select_sample_image("morphology")

    if choice in options:
        # Get additional parameters
        kernel_size = int(input("Enter kernel size (default 3): ") or 3)
        iterations = int(input("Enter iterations (default 1): ") or 1)
        
        out_path = apply_morph_operation(image_path, options[choice], 
                                        kernel_size=kernel_size, 
                                        iterations=iterations)
        print(f"✅ Done. Saved to: {out_path}")
    else:
        print("Invalid choice.")

def run_filters_menu():
    print("\n-- Filter Operations --")
    options = {
        "1": "threshold",
        "2": "adaptive_threshold",
        "3": "gaussian_blur",
        "4": "median_blur",
        "5": "canny",
        "6": "equalize",
        "7": "laplacian",
        "8": "sobel",
        "9": "color_hist_equal",
        "10": "color_sepia",
        "11": "color_sobel"
    }
    for k, v in options.items():
        print(f"{k}. {v.replace('_', ' ').title()}")
    
    choice = input("Select filter (1-11): ")
    image_path = select_sample_image("filters")

    if choice in options:
        # Handle special cases with parameters
        if options[choice] in ["gaussian_blur", "median_blur"]:
            ksize = int(input("Enter kernel size (default 5): ") or 5)
            out_path = apply_filter(image_path, options[choice], ksize=ksize)
        elif options[choice] == "canny":
            t1 = int(input("Enter threshold1 (default 100): ") or 100)
            t2 = int(input("Enter threshold2 (default 200): ") or 200)
            out_path = apply_filter(image_path, options[choice], threshold1=t1, threshold2=t2)
        elif options[choice] == "sobel":
            dx = int(input("Enter dx (default 1): ") or 1)
            dy = int(input("Enter dy (default 1): ") or 1)
            out_path = apply_filter(image_path, options[choice], dx=dx, dy=dy)
        else:
            out_path = apply_filter(image_path, options[choice])
        
        print(f"✅ Done. Saved to: {out_path}")
    else:
        print("Invalid choice.")

def run_transform_menu():
    print("\n-- Geometric Transformations --")
    options = {
        "1": "rotate",
        "2": "scale",
        "3": "translate",
        "4": "flip",
        "5": "affine",
        "6": "perspective",
        "7": "crop"
    }
    for k, v in options.items():
        print(f"{k}. {v.capitalize()}")
    
    choice = input("Select transform (1-7): ")
    image_path = select_sample_image("transform")

    if choice in options:
        if options[choice] == "rotate":
            angle = float(input("Enter rotation angle (default 45): ") or 45)
            out_path = apply_transform(image_path, options[choice], angle=angle)
        elif options[choice] == "scale":
            fx = float(input("Enter x scale factor (default 1.5): ") or 1.5)
            fy = float(input("Enter y scale factor (default 1.5): ") or 1.5)
            out_path = apply_transform(image_path, options[choice], fx=fx, fy=fy)
        elif options[choice] == "translate":
            tx = int(input("Enter x translation (default 50): ") or 50)
            ty = int(input("Enter y translation (default 50): ") or 50)
            out_path = apply_transform(image_path, options[choice], tx=tx, ty=ty)
        elif options[choice] == "flip":
            code = int(input("Enter flip code (0=vertical, 1=horizontal, -1=both, default 1): ") or 1)
            out_path = apply_transform(image_path, options[choice], flip_code=code)
        elif options[choice] == "crop":
            x = int(input("Enter x start (default 100): ") or 100)
            y = int(input("Enter y start (default 100): ") or 100)
            width = int(input("Enter width (default 200): ") or 200)
            height = int(input("Enter height (default 200): ") or 200)
            out_path = apply_transform(image_path, options[choice], x=x, y=y, width=width, height=height)
        else:
            out_path = apply_transform(image_path, options[choice])
        
        print(f"✅ Done. Saved to: {out_path}")
    else:
        print("Invalid choice.")

def run_features_menu():
    print("\n-- Feature Detection --")
    options = {
        "1": "harris_corners",
        "2": "shi_tomasi",
        "3": "sift",
        "4": "surf",
        "5": "orb",
        "6": "blob",
        "7": "contours"
    }
    for k, v in options.items():
        print(f"{k}. {v.replace('_', ' ').title()}")
    
    choice = input("Select feature detection method (1-7): ")
    image_path = select_sample_image("features")

    if choice in options:
        out_path = apply_feature_detection(image_path, options[choice])
        print(f"✅ Done. Saved to: {out_path}")
    else:
        print("Invalid choice.")

def main():
    print("=== Advanced Image Processing CLI Toolkit ===")
    generate_sample_images()  # auto-generate test images

    while True:
        print("\nMain Menu")
        print("1. Morphological Operations")
        print("2. Filter Operations")
        print("3. Geometric Transformations")
        print("4. Feature Detection")
        print("5. Exit")
        choice = input("Choose (1-5): ")

        if choice == "1":
            run_morphology_menu()
        elif choice == "2":
            run_filters_menu()
        elif choice == "3":
            run_transform_menu()
        elif choice == "4":
            run_features_menu()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()