import cv2
from ultralytics import YOLO

# Load the YOLO model
model_path = r"C:\Users\diwak\OneDrive\Desktop\Football\detect\train6\weights\last.pt"
model = YOLO(model_path)

# Open the video file
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame end")
        break

    # Resize frame to model's expected size
    resized_frame = cv2.resize(frame, (640, 450))
    org = cv2.resize(frame, (640, 450))
    # Detect objects in the frame
    result = model.predict(resized_frame,show=True)

    # Display the original frame
    cv2.imshow('original', org)

    if cv2.waitKey(1) == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
