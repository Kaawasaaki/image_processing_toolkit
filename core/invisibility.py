# core/invisibility.py
import cv2
import numpy as np
import time

def apply_invisible_cloak_frame(fg, bg, lower1, upper1, lower2, upper2):
    """
    1) Build a wide HSV mask for red
    2) Enforce a minimum saturation so only “true red” survives
    3) Morphologically CLOSE (fill holes) → OPEN (remove specks) → DILATE (grow)
    4) ERODE (shrink back slightly to avoid halo)
    5) Gaussian-blur to feather into a soft alpha mask
    6) Alpha-blend fg/bg so the red cloth dissolves seamlessly
    """
    # 0) size‐match
    if bg.shape[:2] != fg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_AREA)

    # 1) HSV threshold
    hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, lower1, upper1)
    m2  = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # 2) drop low-saturation (non-vivid red) 
    sat = hsv[:, :, 1]
    mask[sat < 80] = 0

    # 3) big close → open → dilate
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_close, iterations=2)
    se_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  se_open,  iterations=2)
    mask = cv2.dilate(mask, se_open, iterations=3)

    # 4) erode back to avoid a thick halo
    mask = cv2.erode(mask, se_open, iterations=2)

    # 5) feather edges with a large Gaussian
    #    must be odd—tune bigger for softer blend
    blur_ksize = 41  
    soft = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    alpha = (soft.astype(np.float32) / 255.0)[..., None]   # shape H×W×1

    # 6) alpha‐blend
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)
    out = fg_f * (1.0 - alpha) + bg_f * alpha

    return out.astype(np.uint8)
    pass

def run_invisibility_menu():
    print("\n-- Invisibility Cloak --")
    # 1) choose source
    print("1. Webcam")
    print("2. Video file")
    choice = input("Select source (1-2): ")
    use_camera = (choice.strip() == "1")

    # 2) open capture
    src = 0 if use_camera else input("Enter path to video file: ")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("❌ Failed to open source.")
        return

    # 3) grab static background
    print("Capturing background… please step out of view")
    time.sleep(2.0)
    ret, background = cap.read()
    while use_camera and not ret:
        ret, background = cap.read()
    if not ret:
        print("❌ Couldn’t capture background.")
        cap.release()
        return
    if use_camera:
        background = np.flip(background, axis=1)

    # 4) HSV bounds for red
    lower1 = np.array([  0, 120,  70])
    upper1 = np.array([ 10, 255, 255])
    lower2 = np.array([170, 120,  70])
    upper2 = np.array([180, 255, 255])

    print("Press ESC to exit cloak mode.")
    # 5) main loop
    while True:
        ret, frame = cap.read()
        if not ret: break
        if use_camera:
            frame = np.flip(frame, axis=1)

        out = apply_invisible_cloak_frame(
            frame, background,
            lower1, upper1, lower2, upper2
        )

        cv2.imshow("INVISIBLE CLOAK", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
