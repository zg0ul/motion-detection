import cv2
import torch
import time
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression
from utils.plots import Annotator, colors
from utils.general import scale_boxes # Import scale_boxes

weights = "/home/muna/Documents/yolov9_repo/gelan-c-det.pt"
source_image = "/home/muna/Desktop/ppe_0648_jpg.rf.e551a190e852adaae983d2e58b565158.jpg"
output_image = "/home/muna/Desktop/ppe_0648_jpg.rf.e551a190e852adaae983d2e58b565158_yolov9_output.jpg"


start_time = time.time()

# Initialize model
model = DetectMultiBackend(weights)

# Start inference time


# Load and process image
for path, img, img_original, _, _ in LoadImages(source_image, img_size=640):
    img = torch.tensor(img).to(model.device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = non_max_suppression(model(img), 0.25, 0.45)

    # Annotate and save the result
    annotator = Annotator(img_original)

    # Ensure detections exist before processing
    if pred[0] is not None and len(pred[0]) > 0:
        for det in pred[0]:
            print(f"Detection tensor shape: {det.shape}")  # Debugging shape of detections
            if len(det) > 0:
                # If det is 1D (single detection), reshape it to 2D
                if det.ndimension() == 1:
                    det = det.unsqueeze(0)
                print(f"Detections before scaling: {det}")  # Debugging content of detections
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                print(f"Detections after scaling: {det}")  # Debugging content after scaling

                for *xyxy, conf, cls in det:
                    annotator.box_label(xyxy, f"{model.names[int(cls)]} {conf:.2f}", colors(int(cls), True))

    cv2.imwrite(output_image, annotator.result())
    print(f"Saved: {output_image}")
 
# End inference time
print(f"Inference time: {time.time() - start_time:.4f} seconds")