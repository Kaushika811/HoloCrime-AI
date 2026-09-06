import cv2
import os

video_path = "videos/cctv_test.mp4"
output_folder = "output/frames"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened successfully.")
print("FPS:", fps)
print("Total frames:", total_frames)

frame_count = 0
saved_count = 0

# Save 1 frame every 30 frames = approximately 1 frame/second
frame_interval = int(fps)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Save one frame every second
    if frame_count % frame_interval == 0:
        filename = os.path.join(
            output_folder,
            f"frame_{saved_count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        saved_count += 1

    # Display video
    cv2.imshow("HoloCrime AI - Frame Processing", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nProcessing complete.")
print("Frames processed:", frame_count)
print("Frames saved:", saved_count)
print("Saved in:", output_folder)