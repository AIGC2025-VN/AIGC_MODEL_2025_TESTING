import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import json

# --- Config ---
video_path = r"/repoDEEPSORT/congtruong.mp4"
conf_threshold = 0.456789
tracking_class = 5  # hoặc None nếu muốn track tất cả

# --- Init DeepSORT ---
tracker = DeepSort(max_age=30)

# --- Init YOLOv8 model ---
model = YOLO(r"/Construction-PPE-Detection/Model/ppe.pt")

# --- Load class names ---
with open(r"../create data/class_names.json") as f:
    class_dict = json.load(f)  # load đúng JSON
    class_name = [class_dict[str(i)] for i in range(len(class_dict))]  # convert thành list theo thứ tự id


colors = np.random.randint(0, 255, size=(len(class_name), 3))
track = []

# --- Read video ---
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Run detection ---
    results = model(frame)
    result = results[0]

    boxes = result.boxes
    detect = []

    for box in boxes:
        bbox = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        if tracking_class is None:
            if conf < conf_threshold:
                continue
        else:
            if class_id != tracking_class or conf < conf_threshold:
                continue

        x1, y1, x2, y2 = map(int, bbox)
        detect.append([[x1, y1, x2 - x1, y2 - y1], conf, class_id])

    # --- Update tracking ---
    tracks = tracker.update_tracks(detect, frame=frame)

    # --- Draw results ---
    for track_obj in tracks:
        if track_obj.is_confirmed():
            track_id = track_obj.track_id
            ltrb = track_obj.to_tlbr()
            class_id = track_obj.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = f"{class_name[class_id]}-{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # --- Show output ---
    cv2.imshow("OT", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
