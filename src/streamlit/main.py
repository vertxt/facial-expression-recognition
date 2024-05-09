import streamlit as st

models = {
    "efficientnet_b0",
    "efficientnet_b2",
}

face_detectors = {
    "MTCNN",
    "RetinaFace",
    "yolo-face",
}

def image_demo():
    st.title("Image demo")

    with st.sidebar:
        model = st.selectbox("Model", list(models))
        face_detector = st.selectbox("Detector", list(face_detectors))
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5)

def video_demo():
    st.title("Video demo")

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
