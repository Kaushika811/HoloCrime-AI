from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Objects relevant to our surveillance project
relevant_objects = [
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "cell phone"
]

video_path = "videos/cctv_test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully.")
print("Starting HoloCrime AI object detection...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    # Resize for faster CPU processing
    frame = cv2.resize(frame, (640, 360))

    # Run YOLO
    results = model(
        frame,
        imgsz=640,
        verbose=False
    )

    # Get annotated frame
    annotated_frame = frame.copy()

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            # Keep only relevant objects
            if class_name in relevant_objects and confidence >= 0.40:

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Display
    cv2.imshow(
        "HoloCrime AI - Relevant Object Detection",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection finished.")