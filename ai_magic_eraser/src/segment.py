import torch
import cv2 as cv
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

MODEL_PATH = "../models/sam_vit_b_01ec64.pth"


def segment_object(image, point):
    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
    sam.to("cpu")
    predictor = SamPredictor(sam)

    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_point = np.array([point])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    return best_mask