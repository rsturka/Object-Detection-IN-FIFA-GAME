import streamlit as st
import cv2
import math
import tempfile
from ultralytics import YOLO

st.header("Object Detection in FIFA Game")
# Initialize session state for the current page if not already done
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

# Function to change the current page


def navigate_to(page):
    st.session_state['current_page'] = page


# Navigation bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button('Home', on_click=navigate_to, args=('Home',))
with col2:
    st.button('Detection', on_click=navigate_to, args=('Detection',))
with col3:
    st.button('About Us', on_click=navigate_to, args=('About Us',))
with col4:
    st.button('Help', on_click=navigate_to, args=('Help',))  # New Help button

# Function to display the FAQ content


def show_faq():
    st.header('Frequently Asked Questions')
    faqs = [
        {
            "question": "What is YOLOv8 and how does it work?",
            "answer": "YOLOv8 is a state-of-the-art object detection algorithm that identifies and locates objects in images or video streams in real-time. It is designed for speed and accuracy, making it ideal for applications like sports analytics."
        },
        {
            "question": "How can I upload my video for detection?",
            "answer": "Navigate to the 'Detection' tab and use the file uploader to select your video file. The platform currently supports MP4 format and is optimized for short clips to ensure quick processing."
        },
        {
            "question": "Can I download the analyzed video?",
            "answer": "Yes, once the object detection process is complete, a download link will be available, allowing you to save the processed video with detected objects highlighted."
        },
        {
            "question": "Can I download the analyzed video?",
            "answer": "Yes, once the object detection process is complete, a download link will be available, allowing you to save the processed video with detected objects highlighted."
        },
        {
            "question": "What are the system requirements for running this Streamlit app?",
            "answer": "While the app runs on a server and not on your local machine, ensuring a stable internet connection is crucial for smooth video uploads and downloads. The app is optimized for modern web browsers like Chrome, Firefox, and Safari."
        },
        {
            "question": "How accurate is the object detection in videos?",
            "answer": "The accuracy of object detection can vary depending on the video quality, lighting conditions, and the objects' distance from the camera. Our model, YOLOv8, is among the best for handling diverse conditions, but some inaccuracies can still occur."
        },
        {
            "question": "What types of objects can the model detect in a FIFA game?",
            "answer": "The model is trained to detect key elements of a FIFA game, such as the ball, players from different teams, and possibly other significant objects like goalposts. The exact capabilities depend on the model's training data."
        },
        {
            "question": "Can this technology be used for other sports or applications?",
            "answer": "Absolutely! While our current demonstration focuses on FIFA games, the underlying technology is versatile and can be adapted for other sports, security surveillance, wildlife monitoring, and any scenario where object detection is valuable."
        },
        {
            "question": "I encountered an error while uploading a video. What should I do?",
            "answer": "Ensure the video is in MP4 format and does not exceed the size limit. If the issue persists, try refreshing the page or using a different browser. For continuous problems, contact our support team."
        },
        {
            "question": "Is it possible to customize the object detection model for specific needs?",
            "answer": "Yes, the model can be customized or retrained to detect different objects or to improve accuracy on specific video types. This requires additional data and computational resources. Feel free to reach out for more information on custom solutions."
        },
    ]

    for faq in faqs:
        with st.expander(f"Q: {faq['question']}"):
            st.write(f"A: {faq['answer']}")


# Page rendering based on the current_page state
if st.session_state['current_page'] == 'Home':
    st.subheader('What is FIFA Game?')
    st.write('''
        FIFA is one of the most popular and enduring video game franchises in the world, 
        allowing players to experience the excitement of football in a virtual format. 
        It features realistic gameplay, detailed graphics, and licenses with top football leagues 
        and teams around the globe, making it a favorite among fans of the sport.
    ''')
    st.image('https://github.com/Nitin-Diwakar/Object-Detection-in-Video-Streams/blob/master/Image/stadium2.jpg?raw=true')
    st.subheader('Elevate Your Gaming Strategy with Advanced AI')
    st.write('''
        Welcome to a new era of gaming analytics, where advanced artificial intelligence meets the dynamism of football. 
        Our platform harnesses the power of YOLOv8, the latest in object detection technology, to bring you insights 
        like never before. Imagine being able to track every move of the players and the ball with pinpoint accuracy — 
        that's the precision we offer.

        With a foundation built on a robust dataset of thousands of images, meticulously annotated, our AI model 
        doesn't just watch the game — it understands it.

        But what's power without speed? Thanks to OpenCV, our technology doesn't skip a beat. Frame by frame, the action 
        is analyzed, ensuring no detail is missed, no matter how fast the play.

        And the best part? You don't need to be a tech wizard to use it. Streamlit's sleek interface makes interacting 
        with this advanced technology as easy as enjoying the game itself. Whether you're a gamer looking to refine your 
        tactics, a coach aiming to dissect team formations, or just a fan who loves the details, our platform is your 
        new home ground.

        So dive in, explore, and let's take the game to new heights — together. This isn't just about the now; it's 
        about shaping the future of sports gaming and strategy with AI as your ally.

    ''')
    st.subheader('What is YOLOv8?')
    st.write('''
        YOLOv8 was introduced in 2022 by Ultralytics.
        As a cutting-edge, state-of-the-art (SOTA) model, YOLOv8 builds on the success of previous versions, introducing new features and improvements for enhanced performance, flexibility, and efficiency. YOLOv8 supports a full range of vision AI tasks, including detection, segmentation, pose estimation, tracking, and classification. This versatility allows users to leverage YOLOv8's capabilities across diverse applications and domains.
        
        YOLOv8 is an anchor-free model.This means it predicts directly the center of an object instead of the offset from a known anchor box.
        
        Object detection is a task that involves identifying the location and class of objects in an image or video stream.

        The output of an object detector is a set of bounding boxes that enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a scene, but don't need to know exactly where the object is or its exact shape.
    ''')
    st.image('https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png')

elif st.session_state['current_page'] == 'About Us':
    st.header('Contributors')
    contributors = [
        {
            "name": "Sanjana Bafana",
            "linkedin": "https://linkedin.com/in/sanjana-bafana-722720194",
            # Replace with the actual path or URL
            "photo": "https://avatars.githubusercontent.com/u/128772250?v=4"
        },
        {
            "name": "Harshad Gaikwad",
            "linkedin": "https://www.linkedin.com/in/harshad-gaikwad-1157511b4/",
            # Replace with the actual path or URL
            "photo": "https://avatars.githubusercontent.com/u/132460203?v=4"
        },
        {
            "name": "Nitin Diwakar",
            # Replace with the actual LinkedIn URL when available
            "linkedin": "https://www.linkedin.com/in/diwakarnitin/",
            # Replace with the actual path or URL
            "photo": "https://avatars.githubusercontent.com/u/72300414?v=4"
        },
        {
            "name": "Rajbir Singh",
            # Replace with the actual LinkedIn URL when available
            "linkedin": "https://www.linkedin.com/in/rajbir-singh-406b05241/",
            # Replace with the actual path or URL
            "photo": "https://avatars.githubusercontent.com/u/115075229?v=4"
        }
    ]

    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
        }
        .markdown-text-container {
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create a column for each contributor
    for contributor in contributors:
        cols = st.columns([1, 2, 5])  # Adjust the ratio as needed
        with cols[0]:
            # Set the width as per your UI design
            st.image(contributor["photo"], width=100)
        with cols[1]:
            # Additional space added below the subheader
            st.subheader(contributor["name"])
            st.markdown("<br>", unsafe_allow_html=True)
        with cols[2]:
            if contributor["linkedin"]:
                st.markdown(
                    f"[LinkedIn]({contributor['linkedin']})", unsafe_allow_html=True)

elif st.session_state['current_page'] == 'Help':
    show_faq()  # Display the FAQ section when Help is selected
elif st.session_state['current_page'] == 'Detection':
    uploaded_files = st.file_uploader(
        "Choose an MP4 video file", accept_multiple_files=True, type=["mp4"])

    st.warning(
        """
    **NOTE:**
    1. The video should not exceed more than 200MB.
    2. Due to CPU processing, it may take some time. Please wait.
    3. Currently, it is compatible with only these teams: PSG vs. Manchester United.
    4. The video should be in MPEG4 and MP4 format.
    """
    )

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            # label and color for each class
            classNames = ["Ball", "TeamA", "TeamB"]
            # Colors for Ball, TeamA, TeamB respectively
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

            # Load pretrained model
            model_path = "../last.pt"
            model = YOLO(model_path)

            # open video
            cap = cv2.VideoCapture(tfile.name)

            if not cap.isOpened():
                print("Error while reading the frame")

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # Codec used to compress the frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Temporary file for the output video
            output_video_path = tempfile.mktemp(suffix='.mp4')
            frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = cv2.VideoWriter(output_video_path, fourcc,
                                  20.0, (frame_width, frame_height))

            progress_text = "Object Detection in Progress. Please wait..."
            my_bar = st.progress(0)

            frame_processed = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    results = model.predict(frame, stream=True)
                    for r in results:
                        boxes = r.boxes
                        print(r.boxes)
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            cls = int(box.cls[0])
                            class_name = classNames[cls]
                            # Use specific color for each class
                            color = colors[cls]
                            label = f'{class_name} {conf}'
                            text_size = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)[0]
                            c2 = x1 + text_size[0], y1 - text_size[1] - 5

                            # Draw rectangle with class-specific color
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            # Background for text
                            cv2.rectangle(frame, (x1, y1), c2,
                                          color, -1, cv2.LINE_AA)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255, 255, 255], thickness=1,
                                        lineType=cv2.LINE_AA)

                    resize_frame = cv2.resize(
                        frame, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

                    # cv2.imshow("Frame", resize_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    out.write(frame)
                    # Update progress bar
                    frame_processed += 1
                    percent_complete = int(
                        (frame_processed / total_frames) * 100)
                    my_bar.progress(percent_complete, text=progress_text)

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

            # Suggesting users to tag multiple profiles/pages
            st.write(
                "Share your detected video on LinkedIn and don't forget to tag us along with our partners for more visibility!")
            st.markdown("""
            Don't forget to tag us in your post:
            - @Sanjana Bafana
            - @Harshad Gaikwad
            - @Nitin Diwakar
            - @Rajbir Singh
            
            Just type '@' followed by the profile name in your post on LinkedIn to tag us!
            """)

elif st.session_state['current_page'] == 'Help':
    show_faq()
