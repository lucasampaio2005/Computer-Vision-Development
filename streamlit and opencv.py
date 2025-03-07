from PIL import Image
import cv2
import numpy as np
import streamlit as st
import tempfile
from streamlit_option_menu import option_menu

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

with st.sidebar:
    selected = option_menu(
        menu_title = "Menu",
        options=["Detection", "Saved", "About"],
        default_index=0
    )

st.title("Image Detection Web", help=None, anchor=False)
# st.header("Select an image to start the detection", divider="green")
option = st.selectbox(
    "Select an image or video to start the detection",
    ("Image", "Video"),
)
st.write("You selected:", option)



def image_detection():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and option is not None and selected == "Detection":
        image = np.array(Image.open(uploaded_file))
        results = model(image)
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Processed Image Detections", use_container_width=True, clamp=True)

def video_detection():
    uploaded_file = st.file_uploader("Upload Video", ['mp4','mov', 'avi'])
    if uploaded_file is not None and option is not None and selected == "Detection":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())   
        cap = cv2.VideoCapture(tfile.name)
        frame_place_holder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            for result in results:
                for box in result.boxes:
                    cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), thickness=2)
                    cv2.putText(frame, f"{result.names[int(box.cls[0])]}", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), thickness=2)
            annotated_frame = results[0].plot()
            frame_place_holder.image(frame, caption="Processed Video Detections",channels="BGR", use_container_width=True)
            # st.image(frame, channels="BGR", use_column_width=True)
            # st.video(frame)

if(option == 'Image'):
    image_detection()
if(option == 'Video'):
    video_detection()

