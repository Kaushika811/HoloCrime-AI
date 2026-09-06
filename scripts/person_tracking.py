from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

video_path = "videos/walking/walking_01.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully.")
print("Starting person tracking...")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    frame_count += 1

    # Skip every alternate frame
    if frame_count % 3 != 0:
        continue

    # Resize
    frame = cv2.resize(frame, (640, 360))

    # YOLO + ByteTrack
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

    cv2.imshow(
        "HoloCrime AI - Person Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Person tracking stopped.")