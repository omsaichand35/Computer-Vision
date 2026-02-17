import argparse
import torch
import cv2 as cv
import numpy as np
from torchvision import transforms, models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pothole_dataset.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["normal", "pothole"]


def analyze_frame(frame):
    input_img = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_img)
        _, pred = torch.max(output, 1)
        label = classes[pred.item()]

    display_frame = frame.copy()

    if label == "pothole":
        gray = cv.cvtColor(display_frame, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 50, 150)
        contours, _ = cv.findContours(
            edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv.contourArea(cnt)

            if area <= 500:
                continue

            x, y, w, h = cv.boundingRect(cnt)

            if area < 2000:
                severity = "Low"
                color = (0, 255, 0)
            elif area < 5000:
                severity = "Medium"
                color = (0, 255, 255)
            else:
                severity = "High"
                color = (0, 0, 255)

            cv.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(
                display_frame,
                severity,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv.putText(
        display_frame,
        f"Status: {label}",
        (20, 20),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    return display_frame, label

def run_live_camera():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the default camera.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame, _ = analyze_frame(frame)

            cv.imshow("Live Camera", display_frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def run_image(image_path):
    frame = cv.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    display_frame, _ = analyze_frame(frame)

    cv.imshow("Image Test", display_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


def run_camera_snapshot():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the default camera.")

    frame = None
    try:
        while True:
            ret, current = cap.read()
            if not ret:
                break

            cv.imshow("Camera Snapshot", current)
            key = cv.waitKey(1) & 0xFF
            if key == ord("c"):
                frame = current.copy()
                break
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

    if frame is None:
        return

    display_frame, _ = analyze_frame(frame)
    cv.imshow("Snapshot Result", display_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole detection demo")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image for testing"
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture a snapshot from the camera and run detection"
    )
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    elif args.capture:
        run_camera_snapshot()
    else:
        run_live_camera()
