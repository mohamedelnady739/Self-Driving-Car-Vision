import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time
from collections import Counter




st.set_page_config(
    page_title="Self Driving Car Vision",
    page_icon="🚗",
    layout="wide"
)




MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ best.pt not found.\n\n"
        "Put the trained model inside:\n"
        "models/best.pt"
    )

    st.stop()


model = load_model()



CLASS_NAMES = {
    0: "biker",
    1: "car",
    2: "pedestrian",
    3: "trafficLight",
    4: "trafficLight-Green",
    5: "trafficLight-GreenLeft",
    6: "trafficLight-Red",
    7: "trafficLight-RedLeft",
    8: "trafficLight-Yellow",
    9: "trafficLight-YellowLeft",
    10: "truck"
}


VEHICLE_CLASSES = {
    "car",
    "truck"
}




st.sidebar.title("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Mode",
    [
        "Image Detection",
        "Video Detection",
        "Tracking"
    ]
)


confidence = st.sidebar.slider(
    "Confidence",
    0.10,
    1.00,
    0.25,
    0.05
)


iou = st.sidebar.slider(
    "IoU",
    0.10,
    1.00,
    0.70,
    0.05
)


tracker = st.sidebar.selectbox(
    "Tracker",
    [
        "bytetrack.yaml",
        "botsort.yaml"
    ]
)




st.title("🚗 Self Driving Car Vision System")

st.markdown(
    """
    ### YOLOv8s Computer Vision System

    Detect, track and count objects in road scenes.
    """
)

st.divider()




def count_classes(result):

    counts = Counter()

    if result.boxes is None:
        return counts

    for cls in result.boxes.cls:

        class_id = int(cls.item())

        name = CLASS_NAMES.get(
            class_id,
            str(class_id)
        )

        counts[name] += 1

    return counts


def show_dashboard(counts, fps=None):

    st.subheader("📊 Detection Dashboard")

    total = sum(counts.values())

    cols = st.columns(6)

    cols[0].metric(
        "Objects",
        total
    )

    cols[1].metric(
        "🚗 Cars",
        counts["car"]
    )

    cols[2].metric(
        "🚚 Trucks",
        counts["truck"]
    )

    cols[3].metric(
        "🚶 Pedestrians",
        counts["pedestrian"]
    )

    cols[4].metric(
        "🚴 Bikers",
        counts["biker"]
    )

    cols[5].metric(
        "🚦 Traffic Lights",
        counts["trafficLight"]
        + counts["trafficLight-Green"]
        + counts["trafficLight-GreenLeft"]
        + counts["trafficLight-Red"]
        + counts["trafficLight-RedLeft"]
        + counts["trafficLight-Yellow"]
        + counts["trafficLight-YellowLeft"]
    )

    if fps is not None:

        st.metric(
            "FPS",
            f"{fps:.1f}"
        )




if mode == "Image Detection":

    st.header("📷 Image Detection")

    uploaded = st.file_uploader(
        "Upload road image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        results = model.predict(
            np.array(image),
            conf=confidence,
            iou=iou,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Detection")

            st.image(
                annotated,
                use_container_width=True
            )

        counts = count_classes(result)

        show_dashboard(counts)

        if counts:

            st.subheader("📋 Classes")

            st.bar_chart(
                dict(counts)
            )

        result_pil = Image.fromarray(
            annotated
        )

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        )

        result_pil.save(
            temp.name
        )

        with open(temp.name, "rb") as f:

            st.download_button(
                "⬇️ Download Result",
                f,
                file_name="detection_result.png",
                mime="image/png"
            )




elif mode == "Video Detection":

    st.header("🎥 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload road video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_path = input_file.name

        cap = cv2.VideoCapture(
            input_path
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps_video = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps_video <= 0:
            fps_video = 30

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_video,
            (width, height)
        )

        frame_placeholder = st.empty()

        progress = st.progress(0)

        counts = Counter()

        frame_count = 0

        start_time = time.time()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                frame,
                conf=confidence,
                iou=iou,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            writer.write(
                annotated
            )

            current_counts = count_classes(
                result
            )

            counts.update(
                current_counts
            )

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                annotated_rgb,
                channels="RGB",
                use_container_width=True
            )

            frame_count += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_count / total_frames,
                        1.0
                    )
                )

        cap.release()
        writer.release()

        elapsed = time.time() - start_time

        processing_fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )

        st.success(
            "✅ Video processing completed!"
        )

        show_dashboard(
            counts,
            processing_fps
        )

        with open(
            output_path,
            "rb"
        ) as f:

            st.download_button(
                "⬇️ Download Processed Video",
                f,
                file_name="self_driving_result.mp4",
                mime="video/mp4"
            )




elif mode == "Tracking":

    st.header("🎯 Object Tracking")

    uploaded_video = st.file_uploader(
        "Upload video for tracking",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_path = input_file.name

        cap = cv2.VideoCapture(
            input_path
        )

        frame_placeholder = st.empty()

        progress = st.progress(0)

        unique_ids = set()

        vehicle_ids = set()

        pedestrian_ids = set()

        biker_ids = set()

        traffic_ids = set()

        frame_count = 0

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        start_time = time.time()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                conf=confidence,
                iou=iou,
                tracker=tracker,
                persist=True,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes

                if boxes.id is not None:

                    ids = boxes.id.cpu().numpy()

                    classes = boxes.cls.cpu().numpy()

                    for track_id, class_id in zip(
                        ids,
                        classes
                    ):

                        track_id = int(
                            track_id
                        )

                        class_id = int(
                            class_id
                        )

                        class_name = CLASS_NAMES.get(
                            class_id,
                            str(class_id)
                        )

                        unique_ids.add(
                            track_id
                        )

                        if class_name in VEHICLE_CLASSES:

                            vehicle_ids.add(
                                track_id
                            )

                        elif class_name == "pedestrian":

                            pedestrian_ids.add(
                                track_id
                            )

                        elif class_name == "biker":

                            biker_ids.add(
                                track_id
                            )

                        elif "trafficLight" in class_name:

                            traffic_ids.add(
                                track_id
                            )

            annotated = result.plot()

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                annotated_rgb,
                channels="RGB",
                use_container_width=True
            )

            frame_count += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_count / total_frames,
                        1.0
                    )
                )

        cap.release()

        elapsed = time.time() - start_time

        fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )

        st.success(
            "✅ Tracking completed!"
        )



        st.subheader(
            "📊 Tracking Dashboard"
        )

        cols = st.columns(5)

        cols[0].metric(
            "🎯 Total IDs",
            len(unique_ids)
        )

        cols[1].metric(
            "🚗 Vehicles",
            len(vehicle_ids)
        )

        cols[2].metric(
            "🚶 Pedestrians",
            len(pedestrian_ids)
        )

        cols[3].metric(
            "🚴 Bikers",
            len(biker_ids)
        )

        cols[4].metric(
            "🚦 Traffic Lights",
            len(traffic_ids)
        )

        st.metric(
            "⚡ Processing FPS",
            f"{fps:.1f}"
        )




st.divider()

