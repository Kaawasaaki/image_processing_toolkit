import cv2
import numpy as np
import os

os.makedirs("images", exist_ok=True)

img = np.zeros((100, 300), dtype=np.uint8)
cv2.putText(img, 'TEST', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 5, cv2.LINE_AA)

noise = (np.random.rand(100, 300) < 0.05).astype(np.uint8) * 255
noisy_img = cv2.bitwise_or(img, noise)

cv2.imwrite("images/input.jpg", noisy_img)
