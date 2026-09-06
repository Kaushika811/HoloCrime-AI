from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Video
video_path = "videos/walking/walking_01.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Restricted zone coordinates
ZONE_X1 = 360
ZONE_Y1 = 80
ZONE_X2 = 520
ZONE_Y2 = 350

print("Video opened successfully.")
print("Restricted Zone Detection Started...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    # Resize
    frame = cv2.resize(frame, (640, 360))

    # Person detection + tracking
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

    # Get detected boxes
    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            # Get person bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Calculate person's center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check whether person is inside zone
            inside_zone = (
                ZONE_X1 < center_x < ZONE_X2
                and
                ZONE_Y1 < center_y < ZONE_Y2
            )

            if inside_zone:

                cv2.putText(
                    annotated_frame,
                    "ALERT: UNAUTHORIZED ENTRY!",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    # Show result
    cv2.imshow(
        "HoloCrime AI - Restricted Zone Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Restricted Zone Detection Stopped.")