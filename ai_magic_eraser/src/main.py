import cv2 as cv
import numpy as np
from segment import segment_object
from inpaint import inpaint_image_lama

# Load image
image = cv.imread(r"C:\Users\omsai\6th-sem Projects\DL for CV\ai_magic_eraser\data\img_2.jpeg")

if image is None:
    print("Failed to load image.")
    exit()

display = image.copy()
mask = np.zeros(image.shape[:2], dtype=np.uint8)

drawing = False
brush_size = 10


def mouse_callback(event, x, y, flags, param):
    global drawing, mask, display

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(mask, (x, y), brush_size, 255, -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False


cv.namedWindow("Magic Eraser")
cv.setMouseCallback("Magic Eraser", mouse_callback)

print("Paint over object. Press ENTER to erase.")
print("Press C to clear mask. ESC to exit.")

while True:
    display = image.copy()

    # Overlay mask in red
    overlay = display.copy()
    overlay[mask > 0] = (0, 0, 255)
    display = cv.addWeighted(display, 0.7, overlay, 0.3, 0)

    cv.imshow("Magic Eraser", display)

    key = cv.waitKey(1) & 0xFF

    if key == 13:  # ENTER
        break

    elif key == ord('c'):  # Clear mask
        mask[:] = 0

    elif key == 27:  # ESC
        cv.destroyAllWindows()
        exit()

cv.destroyAllWindows()

# Inpaint using LaMa
result = inpaint_image_lama(image, mask)

cv.imshow("Result", result)
cv.imwrite(r"C:\Users\omsai\6th-sem Projects\DL for CV\ai_magic_eraser\outputs\result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
