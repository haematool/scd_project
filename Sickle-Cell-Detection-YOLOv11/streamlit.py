import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st


def predict(model, image, min_conf, display_names, only_elongated):
    """
    Perform predictions on the input image using the given model.

    Args:
        model: The object detection model.
        image: The input image.
        min_conf: Minimum confidence threshold for predictions.
        display_names: Boolean to indicate if class names should be displayed.
        only_elongated: Boolean to process only elongated objects (class_id = 0).

    Returns:
        The image with bounding boxes and optional class labels drawn.
    """
    # Perform prediction
    results = model.predict(source=image)

    # Define class names and colors
    class_names = model.names
    colors = {
        0: (0, 0, 255),  # Elongated
        1: (0, 255, 0),  # Circular
        2: (255, 0, 0)   # Other
    }

    # Iterate through results and draw bounding boxes
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if float(conf) < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)

            # Skip non-elongated objects if only_elongated is True
            if only_elongated and class_id != 0:
                continue

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], 2)

            # Display class name if enabled
            if display_names:
                label = class_names[class_id][:3].upper()
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return image



def main():
    st.title('Sickle Cell Detection')
    st.sidebar.title('SCD Application')
    st.sidebar.subheader('Parameters')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    # # Loading config
    # config_name = './config.yml'
    # config = get_config(config_filepath=config_name)


    # if 'min_conf' in config:
    #     min_conf = config['min_conf']
    # else:
    #     min_conf = config['min_conf'] = 0.4


    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    display_names = st.sidebar.checkbox("Display names")
    only_elongated = st.sidebar.checkbox("Detect only elongated cells")
    min_conf = st.sidebar.slider("Min Confidence", min_value=0.0, max_value=1.0, value=0.4, step=0.1) 

    st.sidebar.caption('Version v1.0')
    st.sidebar.image('logo.png', use_container_width=True)


    if uploaded_file is not None:
        # Convert the uploaded file to a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Decode the image using OpenCV
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, channels="BGR")
        
        # stored_config = get_config(config_filepath=config_name)

        model = YOLO("best.pt")

        pred_img = predict(model, image, min_conf, display_names, only_elongated)

        st.image(pred_img, channels = "BGR")



if __name__ == "__main__":
    main()