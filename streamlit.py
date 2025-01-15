import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from unet_model_architecture import UNet
import torch
from torchvision import transforms
from PIL import Image
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


def predict_unet(model, image):

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor()
    ])
    input_tensor = preprocess(Image.fromarray(image)).unsqueeze(0)  

    
    with torch.no_grad():
        output = model(input_tensor)  
        predicted_mask = torch.sigmoid(output).squeeze(0).numpy()  # Shape: (4, H, W)

    
    combined_mask = np.zeros((512, 512), dtype=np.uint8)

    #class 2 are the elongated witch are white the rest are gray, the 0 are background

    combined_mask[predicted_mask[1] > 0.5] = 127  
    combined_mask[predicted_mask[2] > 0.5] = 255  
    combined_mask[predicted_mask[3] > 0.5] = 127  
    
    mask_image = cv2.merge([combined_mask, combined_mask, combined_mask])
    return mask_image





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
    
    
    st.sidebar.title("Select_model")
    model_type = st.sidebar.selectbox("Choose a model", ["YOLO", "U-Net"])
    st.sidebar.caption('Version v1.0')
    st.sidebar.image('logo.png', use_container_width=True)
   

    if uploaded_file is not None:
        # Convert the uploaded file to a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Decode the image using OpenCV
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, channels="BGR")
        
        # stored_config = get_config(config_filepath=config_name)
        if model_type == "YOLO":
            model = YOLO("best.pt")

            pred_img = predict(model, image, min_conf, display_names, only_elongated)

            st.image(pred_img, channels = "BGR")

        if model_type == "U-Net":
        
            model = UNet(num_classes=4)  
            state_dict = torch.load("cell_hybridloss_100_epoch.pth", map_location=torch.device('cpu'))

            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval() 
            pred_img = predict_unet(model,image)
            resized_img = cv2.resize(pred_img, ((image.shape[1]), image.shape[0]))
            st.image(resized_img, channels = "BGR")
if __name__ == "__main__":
    main()