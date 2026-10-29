import cv2

# ============================================================
# Configuration
# ============================================================

VIDEO_PATH = "datasets/egocentric_video/cup_lifting_01.mp4"
TARGET_SIZE = (512, 512)


# ============================================================
# Read first frame
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read first frame from video.")

orig_h, orig_w = frame.shape[:2]

print(f"Original frame: {orig_w} x {orig_h}")


# ============================================================
# Apply SAME letterbox operation as your CoTracker code
# ============================================================

max_dim = max(orig_w, orig_h)

pad_top = (max_dim - orig_h) // 2
pad_bottom = max_dim - orig_h - pad_top

pad_left = (max_dim - orig_w) // 2
pad_right = max_dim - orig_w - pad_left

padded = cv2.copyMakeBorder(
    frame,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
    cv2.BORDER_CONSTANT,
    value=[30, 30, 30],
)


# ============================================================
# Resize to CoTracker input resolution
# ============================================================

resized = cv2.resize(padded, TARGET_SIZE)

H, W = resized.shape[:2]

print(f"CoTracker frame: {W} x {H}")


# ============================================================
# Select cup / tumbler with mouse
# ============================================================

print()
print("Draw a tight rectangle around the CUP / TUMBLER.")
print("Press ENTER or SPACE when finished.")
print("Press C to cancel.")
print()

x, y, w, h = cv2.selectROI(
    "Select Cup / Tumbler ROI",
    resized,
    showCrosshair=True,
    fromCenter=False,
)

cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise RuntimeError("No ROI selected.")


# ============================================================
# Convert OpenCV ROI format
#
# selectROI returns:
#
#     (x, y, width, height)
#
# Your CoTracker code wants:
#
#     (x0, y0, x1, y1)
# ============================================================

x0 = int(x)
y0 = int(y)

x1 = int(x + w)
y1 = int(y + h)


# ============================================================
# Calculate normalized coordinates
#
# These are useful because your existing code uses:
#
#     W * 0.42
#     H * 0.38
#     ...
# ============================================================

x0_norm = x0 / W
y0_norm = y0 / H

x1_norm = x1 / W
y1_norm = y1 / H


# ============================================================
# Print results
# ============================================================

print("\n==========================================")
print("CUP / TUMBLER ROI")
print("==========================================")

print("\nPixel coordinates:")
print(f"x0 = {x0}")
print(f"y0 = {y0}")
print(f"x1 = {x1}")
print(f"y1 = {y1}")

print("\nROI:")
print(f"glass_roi = ({x0}, {y0}, {x1}, {y1})")


print("\nNormalized coordinates:")
print(f"x0 = {x0_norm:.4f}")
print(f"y0 = {y0_norm:.4f}")
print(f"x1 = {x1_norm:.4f}")
print(f"y1 = {y1_norm:.4f}")


print("\nPaste this into your CoTracker code:")
print(
    f"glass_roi = ("
    f"int(W * {x0_norm:.4f}), "
    f"int(H * {y0_norm:.4f}), "
    f"int(W * {x1_norm:.4f}), "
    f"int(H * {y1_norm:.4f})"
    f")"
)


# ============================================================
# Show selected ROI for final verification
# ============================================================

preview = resized.copy()

cv2.rectangle(
    preview,
    (x0, y0),
    (x1, y1),
    (0, 255, 0),
    2,
)

cv2.putText(
    preview,
    "CUP / TUMBLER ROI",
    (x0, max(20, y0 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 255, 0),
    2,
)

cv2.imshow("Selected ROI - Press any key", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()