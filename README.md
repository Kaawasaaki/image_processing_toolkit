# Image Processing CLI Toolkit

A command-line image processing toolkit with 30+ operations for educational use and computer vision projects.


---



## 📝 Features

- **Morphological Operations**  
  Erosion, dilation, opening, closing, gradient, top-hat, black-hat, etc.

- **Image Filters**  
  Thresholding (global & adaptive), Gaussian/median blur, Canny, Laplacian, Sobel, histogram equalization, sepia, color Sobel, etc.

- **Geometric Transforms**  
  Rotation, scaling, translation, flip, affine warp, perspective warp, crop.

- **Feature Detection**  
  Harris & Shi–Tomasi corners, SIFT, SURF, ORB, blob detection, contours.

- **Performance Metrics**  
  PSNR, SSIM, MSE with side-by-side visual comparisons.

- **Invisibility Cloak**  
  Real-time “Harry Potter cloak” effect on a single-color cloth, using HSV masking + morphology + alpha blending. Works on webcam or video file directly from the CLI.

- **Educational Focus**  
  Interactive menus, auto-generated sample images in `images/`, parameter tuning and instant feedback.


---

## 🚀 Prerequisites

- Python 3.8+
- `pip`

---


## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Kaawasaaki/image_processing_toolkit.git
   cd image_processing_toolkit

2. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

2. **Run**  

   To run as CLI:

   'python main.py'

   

   To run in GUI:

   'python main.py --mode gui'

   
