import cv2
import numpy as np
import math
from ultralytics import YOLO

# label and color for each class
classNames = ["Ball", "TeamA", "TeamB"]
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Colors for Ball, TeamA, TeamB respectively

# Load pretrained model
model_path = r"C:\Users\diwak\OneDrive\Desktop\Football\detect\train6\weights\last.pt"
model = YOLO(model_path)

# open video
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error while reading the frame")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Statistical Analysis
class_counters = {name: 0 for name in classNames}

# Heatmap Setup
heatmap_matrix = np.zeros((frame_height, frame_width), dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                color = colors[cls]  # Use specific color for each class
                label = f'{class_name} {conf}'
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 5

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Draw rectangle with class-specific color
                cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)  # Background for text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255, 255, 255], thickness=1,
                            lineType=cv2.LINE_AA)

                # Update Statistical Analysis
                class_counters[class_name] += 1

                # Update Heatmap Data
                heatmap_matrix[y1:y2, x1:x2] += 1

        resize_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

        cv2.imshow("Frame", resize_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# Normalize and convert the heatmap matrix to a visible format
heatmap_norm = cv2.normalize(heatmap_matrix, None, 0, 255, cv2.NORM_MINMAX)
heatmap_uint8 = np.uint8(heatmap_norm)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# Display the heatmap
cv2.imshow("Heatmap", heatmap_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print statistical analysis results
print("Detection Counts:", class_counters)
