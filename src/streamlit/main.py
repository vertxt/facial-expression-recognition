import time

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
from torchvision import transforms

from face_detector import FaceDetector

# Settings --------------------------------------------------------------------
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

MODELS_DIR = "../../models"

IMG_SIZE = 224

# Helpers ---------------------------------------------------------------------

class EmotionLabel:
    labels = [
        "Surprise",
        "Fear",
        "Disgust",
        "Happiness",
        "Sadness",
        "Anger",
        "Neutral"
    ]
    
    @staticmethod
    def get_label(index):
        return EmotionLabel.labels[index]
    
    @staticmethod
    def get_index(label):
        return EmotionLabel.labels.index(label)

def preprocess(img):
    test_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    img_tensor = test_transforms(img)
    img_tensor.unsqueeze_(0)

    return img_tensor

# Demo ------------------------------------------------------------------------

models = {
    "enet_b0_7",
    "efficientnet_b0_robust_softmax",
}

face_detectors = {
    "mtcnn",
    "retinaface",
    "yolo",
    "dlib",
}

def image_demo():
    st.title("Image demo")
    st.write(f"CUDA available? {use_cuda}")

    with st.sidebar:
        model_name = st.selectbox("Model", list(models))
        face_detector_type = st.selectbox("Face Detector", list(face_detectors))
        detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5)
        align_face = st.checkbox("Align detected faces", value=False)

    uploaded_files = st.file_uploader("Uploaded image...", type=["png", "jpg"], accept_multiple_files=True)
    if uploaded_files is not None:
        cols = st.columns(len(uploaded_files))

        for i, uploaded_file in enumerate(uploaded_files):
            # Reading the image using cv2.imdecode will create problems (e.g., unexpected number of detected faces)
            #   file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            #   img = cv2.imdecode(file_bytes, 1)

            img = Image.open(uploaded_file)
            img = np.array(img)
            out_img = img.copy()

            face_detector = FaceDetector(face_detector_type)

            fd_start_time = time.time()

            detected_faces = face_detector.detect_faces(img)
            bboxes = face_detector.get_bboxes(detected_faces, detection_threshold)

            fd_end_time = time.time()

            emotion_counts = {label: 0 for label in EmotionLabel.labels}

            fer_start_time = time.time()

            emotions = []
            cropped_faces = []
            for bbox in bboxes:
                x, y, w, h = bbox

                cropped_face = img[y:y+h, x:x+w, :]
                cropped_faces.append(cropped_face)

                model = torch.load(f"./models/raf/{model_name}.pt", map_location=torch.device("cpu"))
                model = model.to(device)
                model.eval()

                img_tensor = preprocess(Image.fromarray(cropped_face))

                scores = model(img_tensor.to(device))
                scores = scores[0].data.cpu().numpy()

                emotion = EmotionLabel.get_label(np.argmax(scores))
                emotions.append(emotion)
                emotion_counts[emotion] += 1

                cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)           
                cv2.putText(out_img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            fer_end_time = time.time()

            with cols[i]:
                st.image(out_img)
                st.write(f"Face detection runtime: {format(fd_end_time - fd_start_time, '.2f')} seconds")
                st.write(f"FER runtime: {format(fer_end_time - fer_start_time, '.2f')} seconds")
                st.write(f"Detected faces: {len(bboxes)}")

                fig, ax = plt.subplots()
                ax.bar(emotion_counts.keys(), emotion_counts.values())
                ax.set_xlabel("Expressions")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                for j, bbox in enumerate(bboxes):
                    face_container = st.container()
                    with face_container:
                        st.image(cropped_faces[j], caption=f"Face {j+1}")
                        st.write(f"Emotion: {emotions[j]}")

def video_demo():
    st.title("Video demo")
    st.write(f"CUDA available? {use_cuda}")

    with st.sidebar:
        model = st.selectbox("Model", list(models))
        face_detector = st.selectbox("Detector", list(face_detectors))
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
        frame_skip = st.number_input("#Frames to skip", value=0)

apps = {
    "image_demo",
    "video_demo"
}

with st.sidebar:
    app = st.selectbox("Demo", list(apps))

if app == "image_demo":
    image_demo()
elif app == "video_demo":
    video_demo()
