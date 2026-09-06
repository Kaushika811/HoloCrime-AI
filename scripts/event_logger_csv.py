from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
video_path = "videos/walking/walking_01.mp4"

# Output folders
screenshot_folder = "output/events"
video_folder = "output/event_videos"
log_folder = "output/logs"

os.makedirs(screenshot_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# CSV log file
csv_path = os.path.join(log_folder, "event_log.csv")

# Create CSV and header
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Event ID",
            "Person ID",
            "Event Type",
            "Frame Number",
            "Timestamp",
            "Screenshot",
            "Event Video"
        ])

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Restricted zone
ZONE_X1 = 360
ZONE_Y1 = 80
ZONE_X2 = 520
ZONE_Y2 = 350

person_was_inside = False
event_count = 0
frame_number = 0

video_writer = None
recording_event = False

current_event_data = None

print("HoloCrime AI CSV Event Logging Started...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    frame_number += 1

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    # Person detection and tracking
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
    detected_person_id = "Unknown"

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Get tracking ID
            if box.id is not None:
                detected_person_id = int(box.id[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            inside_zone = (
                ZONE_X1 < center_x < ZONE_X2
                and
                ZONE_Y1 < center_y < ZONE_Y2
            )

            if inside_zone:

                person_inside_now = True

                cv2.putText(
                    annotated_frame,
                    "ALERT: UNAUTHORIZED ENTRY!",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    # EVENT START
    if person_inside_now and not person_was_inside:

        event_count += 1

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        # Screenshot
        screenshot_name = (
            f"event_{event_count}_unauthorized_{timestamp}.jpg"
        )

        screenshot_path = os.path.join(
            screenshot_folder,
            screenshot_name
        )

        cv2.imwrite(
            screenshot_path,
            annotated_frame
        )

        # Video
        video_name = (
            f"event_{event_count}_unauthorized_{timestamp}.mp4"
        )

        video_path_output = os.path.join(
            video_folder,
            video_name
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            video_path_output,
            fourcc,
            fps,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        recording_event = True

        current_event_data = {
            "event_id": event_count,
            "person_id": detected_person_id,
            "frame": frame_number,
            "timestamp": timestamp,
            "screenshot": screenshot_path,
            "video": video_path_output
        }

        print("\nEVENT DETECTED!")
        print("Event ID:", event_count)
        print("Person ID:", detected_person_id)
        print("Frame:", frame_number)

    # Record video
    if recording_event and video_writer is not None:
        video_writer.write(annotated_frame)

    # EVENT END
    if recording_event and not person_inside_now:

        video_writer.release()

        # Save event to CSV
        with open(csv_path, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                current_event_data["event_id"],
                current_event_data["person_id"],
                "Unauthorized Entry",
                current_event_data["frame"],
                current_event_data["timestamp"],
                current_event_data["screenshot"],
                current_event_data["video"]
            ])

        print("Event video saved.")
        print("Event logged in CSV.")

        video_writer = None
        recording_event = False
        current_event_data = None

    person_was_inside = person_inside_now

    cv2.imshow(
        "HoloCrime AI - Event Logging",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()

print("\nSystem stopped.")
print("Total events:", event_count)
print("CSV Log:", csv_path)