import torch
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "big-lama.pt"

device = "cpu"

print("Loading LaMa model on CPU...")
model = torch.jit.load(str(MODEL_PATH), map_location=device)
model.eval()
print("LaMa loaded successfully.")


def pad_to_multiple_of_8(image, mask):
    h, w = image.shape[:2]

    new_h = (h + 7) // 8 * 8
    new_w = (w + 7) // 8 * 8

    pad_h = new_h - h
    pad_w = new_w - w

    image_padded = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )

    mask_padded = cv2.copyMakeBorder(
        mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )

    return image_padded, mask_padded, h, w


def inpaint_image_lama(image, mask):
    # Pad to valid size
    image, mask, orig_h, orig_w = pad_to_multiple_of_8(image, mask)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_rgb = image_rgb.astype("float32") / 255.0
    mask = mask.astype("float32") / 255.0

    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor, mask_tensor)

    output = output[0].permute(1, 2, 0).cpu().numpy()
    output = (output * 255).astype("uint8")
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Crop back to original size
    output = output[:orig_h, :orig_w]

    return output
