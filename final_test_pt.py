import os
from dotenv import load_dotenv
import cv2
import time
from ultralytics import YOLO

# Load .env
load_dotenv()


def draw_text_with_background(frame, text, position, font_scale=0.6, color=(255, 255, 255), thickness=1,
                              bg_color=(0, 0, 0), alpha=0.7, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    x, y = position

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - padding, y - text_height - padding), (x + text_width + padding, y + padding), bg_color,
                  -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def point_in_box(center, box):
    cx, cy = center
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def main():
    # Load YOLO model
    model = YOLO("Model/ppe.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    # Set webcam resolution to 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Verify resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != 1280 or height != 720:
        print(f"Warning: Could not set resolution to 1280x720. Current resolution: {width}x{height}")

    # Create window
    cv2.namedWindow("YOLOv8 Annotated Feed", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip frame horizontally to correct mirror effect
        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]

        # Define monitoring zone (40% width, 80% height, centered)
        box_width = int(frame_width * 0.4)
        box_height = int(frame_height * 1)
        x1 = (frame_width - box_width) // 2
        y1 = int(frame_height * 0.2)  # Slightly offset from top
        x2 = x1 + box_width
        y2 = y1 + box_height
        monitor_zone = (x1, y1, x2, y2)

        # Run YOLO inference
        results = model(frame)

        hardhat_boxes = []
        vest_boxes = []
        person_boxes = []

        # Process detection results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cls = int(box.cls[0])

                    if model.names[cls] == "Hardhat":
                        hardhat_boxes.append((x1, y1, x2, y2))
                    elif model.names[cls] == "Safety Vest":
                        vest_boxes.append((x1, y1, x2, y2))
                    elif model.names[cls] == "Person":
                        person_boxes.append((x1, y1, x2, y2))

        # Initialize zone status
        zone_status = "Khong co nguoi"
        zone_color = (0, 0, 255)  # Red by default
        message = ""

        # Check for people in the monitoring zone
        for px1, py1, px2, py2 in person_boxes:
            center_x = (px1 + px2) // 2
            center_y = (py1 + py2) // 2
            person_box = (px1, py1, px2, py2)

            if point_in_box((center_x, center_y), monitor_zone):
                has_hardhat = any(boxes_overlap(person_box, h) for h in hardhat_boxes)
                has_vest = any(boxes_overlap(person_box, v) for v in vest_boxes)

                if has_hardhat and has_vest:
                    zone_status = "Full PPE Detected"
                    zone_color = (0, 255, 0)  # Green
                else:
                    zone_status = "Missing Equipment"
                    zone_color = (0, 0, 255)  # Red
                    if not has_hardhat:
                        message += "Thieu mu. "
                    if not has_vest:
                        message += "Thieu ao bao ho. "
                break  # Process only one person

        # Draw monitoring zone
        cv2.rectangle(frame, monitor_zone[:2], monitor_zone[2:], zone_color, 3)
        draw_text_with_background(frame, zone_status, (monitor_zone[0] + 5, monitor_zone[1] - 10), bg_color=zone_color)

        # Draw warning message if applicable
        if message:
            draw_text_with_background(frame, message, (10, 30), font_scale=0.6, bg_color=(0, 0, 255))

        # Display frame
        cv2.imshow("YOLOv8 Annotated Feed", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()