from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load YOLO model
model = YOLO("yolo11n.pt")

# Video path
video_path = "videos/walking/walking_01.mp4"

# Output folders
output_folder = "output/events"
video_folder = "output/event_videos"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video information
fps = cap.get(cv2.CAP_PROP_FPS)

# Output frame size
frame_width = 640
frame_height = 360

# Restricted zone
ZONE_X1 = 360
ZONE_Y1 = 80
ZONE_X2 = 520
ZONE_Y2 = 350

# Event variables
person_was_inside = False
event_count = 0
frame_number = 0

# Video writer
video_writer = None
recording_event = False

print("HoloCrime AI Event Video Detection Started...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    frame_number += 1

    # Resize frame
    frame = cv2.resize(
        frame,
        (frame_width, frame_height)
    )

    # YOLO person detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.50,
        imgsz=640,
        verbose=False
    )

    # Draw detection results
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

    # Check if person is inside
    person_inside_now = False

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check zone
            inside_zone = (
                ZONE_X1 < center_x < ZONE_X2
                and
                ZONE_Y1 < center_y < ZONE_Y2
            )

            if inside_zone:

                person_inside_now = True

                # Display alert
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

        # Save screenshot
        screenshot_filename = (
            f"event_{event_count}_"
            f"unauthorized_entry_"
            f"{timestamp}.jpg"
        )

        screenshot_path = os.path.join(
            output_folder,
            screenshot_filename
        )

        cv2.imwrite(
            screenshot_path,
            annotated_frame
        )

        # Create event video
        video_filename = (
            f"event_{event_count}_"
            f"unauthorized_entry_"
            f"{timestamp}.mp4"
        )

        video_path_output = os.path.join(
            video_folder,
            video_filename
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            video_path_output,
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        recording_event = True

        print("\nEVENT DETECTED!")
        print("Event:", event_count)
        print("Type: Unauthorized Entry")
        print("Frame:", frame_number)
        print("Screenshot:", screenshot_path)
        print("Video recording started:", video_path_output)

    # RECORD EVENT VIDEO
    if recording_event and video_writer is not None:

        video_writer.write(
            annotated_frame
        )

    # EVENT END
    if recording_event and not person_inside_now:

        print("Person left restricted zone.")
        print("Event video saved.")

        video_writer.release()

        video_writer = None
        recording_event = False

    # Update previous status
    person_was_inside = person_inside_now

    # Display video
    cv2.imshow(
        "HoloCrime AI - Event Video Detection",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()

print("\nHoloCrime AI Event Video Detection Stopped.")
print("Total events detected:", event_count)