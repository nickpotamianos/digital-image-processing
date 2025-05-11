import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

# Load and prepare the image
img_path = os.path.join("Images", "Ασκηση 6", "hallway.png")
img_rgb = Image.open(img_path).convert("RGB")
img = np.array(img_rgb)
gray = rgb2gray(img)

# --- 1. Edge detection -------------------------------------------------
# Slightly higher sigma to reduce texture noise; tweak low/high thresholds
edges = canny(gray, sigma=2.0, low_threshold=0.05, high_threshold=0.2)

# --- 2. Probabilistic Hough Transform ----------------------------------
lines = probabilistic_hough_line(
    edges,
    threshold=10,     # accumulator threshold
    line_length=60,   # minimum accepted line length
    line_gap=10       # maximum gap to allow connecting segments
)

# --- 3. Overlay detected lines on the original image -------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img)
for ((x0, y0), (x1, y1)) in lines:
    ax.plot([x0, x1], [y0, y1], linewidth=2)  # default color cycle
ax.set_axis_off()
ax.set_title(f"Probabilistic Hough lines detected: {len(lines)}")

# Save the overlay for downloading
overlay_path = "hallway_hough.png"
fig.savefig(overlay_path, bbox_inches="tight")
overlay_path
