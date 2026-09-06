from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load YOLO model
model = YOLO("yolo11n.pt")

# Video path
video_path = "videos/walking/walking_01.mp4"

# Output folder
output_folder = "output/events"

# Create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Restricted zone
ZONE_X1 = 360
ZONE_Y1 = 80
ZONE_X2 = 520
ZONE_Y2 = 350

# Event status
person_was_inside = False
event_count = 0
frame_number = 0

print("HoloCrime AI Event Detection Started...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    frame_number += 1

    # Resize frame
    frame = cv2.resize(frame, (640, 360))

    # Track persons
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.50,
        imgsz=640,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Draw restricted zone
    cv2.rectangle(
        annotated_frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "RESTRICTED ZONE",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    person_inside_now = False

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Person center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check restricted zone
            inside_zone = (
                ZONE_X1 < center_x < ZONE_X2
                and ZONE_Y1 < center_y < ZONE_Y2
            )

            if inside_zone:

                person_inside_now = True

                # Show alert
                cv2.putText(
                    annotated_frame,
                    "ALERT: UNAUTHORIZED ENTRY!",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    # Detect first entry into zone
    if person_inside_now and not person_was_inside:

        event_count += 1

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"event_{event_count}_"
            f"unauthorized_entry_"
            f"{timestamp}.jpg"
        )

        filepath = os.path.join(
            output_folder,
            filename
        )

        # Save evidence screenshot
        cv2.imwrite(
            filepath,
            annotated_frame
        )

        print("\nEVENT DETECTED!")
        print("Event:", event_count)
        print("Type: Unauthorized Entry")
        print("Frame:", frame_number)
        print("Evidence saved:", filepath)

    # Update previous state
    person_was_inside = person_inside_now

    # Show video
    cv2.imshow(
        "HoloCrime AI - Event Detection",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\nHoloCrime AI Event Detection Stopped.")
print("Total events detected:", event_count)