import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8s.pt'):
        """
        YOLO Safe Initialization.
        - model.to() is NOT called manually
        - Ultralytics handles device automatically
        - yolov8s.pt will be auto-downloaded if missing
        """
        self.model = YOLO(model_path)

    def class_name(self, cls_id: int) -> str:
        """Human-readable COCO / model class name for UI and voice when memory has no match."""
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(int(cls_id), f"class_{cls_id}"))
        if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
            return str(names[int(cls_id)])
        return f"object_{cls_id}"

    def detect_and_crop(
        self,
        image,
        conf: float = 0.6,
        iou: float = 0.45,
        max_det: int = 10,
        min_area_abs: int = 1000,
    ):
        """
        Run inference with NMS and post-filters.
        Live video defaults (conf=0.6, min_area_abs=1000) reduce chatter.
        For still photos, callers typically pass lower conf (e.g. 0.35–0.45) and scale-aware min_area_abs.
        """
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
        
        raw_boxes = []
        img_h, img_w, _ = image.shape

        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                box_conf = float(box.conf[0].item())
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = [int(c) for c in coords]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                raw_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': box_conf,
                    'class': cls
                })

        # 2. REMOVE SMALL / NOISY BOXES & OVERLAPPING DUPLICATES
        # Sort by confidence high to low
        sorted_boxes = sorted(raw_boxes, key=lambda x: x['conf'], reverse=True)
        final_detections = []
        
        for cur in sorted_boxes:
            x1, y1, x2, y2 = cur['bbox']
            area = (x2 - x1) * (y2 - y1)
            
            # Filter by area
            if area < min_area_abs:
                continue
            
            # 3. REMOVE OVERLAPPING DUPLICATE BOXES (IoU > 0.7)
            is_duplicate = False
            for kept in final_detections:
                overlap = self._calculate_iou(cur['bbox'], kept['bbox'])
                if overlap > 0.7:
                    # Optional Class-aware filter
                    if cur['class'] == kept['class']:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                crop = image[y1:y2, x1:x2]
                final_detections.append({
                    'bbox': [x1, y1, x2, y2], 
                    'crop': crop,
                    'det_conf': cur['conf'],
                    'class': cur['class']
                })

        return final_detections

    def _calculate_iou(self, box1, box2):
        """Standard Intersection over Union calculation."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def draw_bbox(self, image, bbox, label="Object", color=(255, 60, 255)): # Default to neon pink if none
        x1, y1, x2, y2 = bbox
        
        # Cyberpunk Neon Glow effect (Multiple rectangles with decreasing thickness)
        glow_color = (color[0], color[1], color[2]) # BGR
        cv2.rectangle(image, (x1, y1), (x2, y2), glow_color, 6)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1) # core white line
        
        # Anime HUD Style corner brackets
        length = 20
        thick = 3
        cv2.line(image, (x1, y1), (x1+length, y1), color, thick)
        cv2.line(image, (x1, y1), (x1, y1+length), color, thick)
        cv2.line(image, (x2, y1), (x2-length, y1), color, thick)
        cv2.line(image, (x2, y1), (x2, y1+length), color, thick)
        cv2.line(image, (x1, y2), (x1+length, y2), color, thick)
        cv2.line(image, (x1, y2), (x1, y2-length), color, thick)
        cv2.line(image, (x2, y2), (x2-length, y2), color, thick)
        cv2.line(image, (x2, y2), (x2, y2-length), color, thick)

        # TARGET LOCKED text formatting
        if label != "Object":
            hud_label = f"TARGET LOCKED: {label.upper()}"
        else:
            hud_label = f"TARGET LOCKED: UNKNOWN"

        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.3
        thickness = 2
        (w, h), _ = cv2.getTextSize(hud_label, font, font_scale, thickness)
        
        label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
        cv2.rectangle(image, (x1, label_y - h - 5), (x1 + w + 10, label_y + 5), color, -1)
        
        # Black text on neon background for intense contrast
        cv2.putText(image, hud_label, (x1 + 5, label_y), font, font_scale, (0, 0, 0), thickness)
        
        return image
