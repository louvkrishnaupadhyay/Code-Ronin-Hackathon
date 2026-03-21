import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        YOLO Safe Initialization.
        - model.to() is NOT called manually
        - Ultralytics handles device automatically
        - yolov8n.pt will be auto-downloaded if missing
        """
        self.model = YOLO(model_path)

    def detect_and_crop(self, image):
        """Run inference. Device is managed internally by Ultralytics."""
        results = self.model(image, verbose=False)
        detections = []
        img_h, img_w, _ = image.shape

        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(c) for c in coords]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                crop = image[y1:y2, x1:x2]
                detections.append({'bbox': [x1, y1, x2, y2], 'crop': crop})
        return detections

    def draw_bbox(self, image, bbox, label="Object", color=(0, 255, 0)):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image
