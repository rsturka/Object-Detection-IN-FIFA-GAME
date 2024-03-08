import streamlit as st
import cv2
import math
import tempfile
from ultralytics import YOLO


st.header("Object Detection In FIFA Game")

uploaded_files = st.file_uploader("Choose an MP4 video file", accept_multiple_files=True, type=["mp4"])

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())


        # label and color for each class
        classNames = ["Ball", "TeamA", "TeamB"]
        colors = [(0, 255, 0),(0, 0, 255),(255, 0, 0)]  # Colors for Ball, TeamA, TeamB respectively

        # Load pretrained model
        model_path = "last.pt"
        model = YOLO(model_path)

        # open video
        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            print("Error while reading the frame")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
        output_video_path = tempfile.mktemp(suffix='.mp4')  # Temporary file for the output video
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

        
        progress_text = "Object Detection in Progress. Please wait..."
        my_bar = st.progress(0)

        frame_processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = model.predict(frame,stream=True)
                for r in results:
                    boxes = r.boxes
                    print(r.boxes)
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

                resize_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

                #cv2.imshow("Frame", resize_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                out.write(frame)
                # Update progress bar
                frame_processed += 1
                percent_complete = int((frame_processed / total_frames) * 100)
                my_bar.progress(percent_complete,text=progress_text)

            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        my_bar.empty()  # Clear the progress bar
        
        # Display download button
        with open(output_video_path, 'rb') as file:
            st.download_button(label="Download Processed Video",
                               data=file,
                               file_name="processed_video.mp4",
                               mime='video/mp4')
