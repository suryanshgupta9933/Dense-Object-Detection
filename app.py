import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import time
import os
import random

# Import your custom YOLOv5 model
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Define the path to your saved YOLOv5 model
model_path = "weights/best.pt"

# Set the device to GPU if available
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# Define a function to draw bounding boxes on the image
def draw_bounding_boxes(image, boxes, confidences, class_ids):
    class_names = ['object'] # Replace with your own class names
    
    # Loop over all the detections
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        class_name = class_names[class_id]
        
        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # label = f"{class_name}: {confidence:.2f}"
        # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
    
    return image

# Define the Streamlit app
def main():
    # Load the model
    model = attempt_load(model_path, device=device)
    st.set_page_config(page_title="Yolov5-SKU110", page_icon="💡")
    st.title("🔪Yolov5-SKU110K")
    st.write("Object Detection in Dense Environments using Yolov5 on SKU110K dataset")
    conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_thres = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file)
        shape = image.size
        image = image.resize((640, 640))
        img = np.array(image)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        # Run the YOLOv5 model on the image
        pred = model(img)[0] 
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
        # convert to numpy
        pred = [x.detach().cpu().numpy() for x in pred]
        # convert to int
        pred = [x.astype(int) for x in pred]
        # Post-process the output and draw bounding boxes on the image
        boxes = []
        confidences = []    
        class_ids = []
        for det in pred:
            if det is not None and len(det):
                # Scale the bounding box coordinates to the original image size
                det[:, :4] = det[:, :4] / 640 * image.size[0]
                for *xyxy, conf, cls in det:
                    boxes.append(xyxy)
                    confidences.append(conf.item())
                    class_ids.append(int(cls.item()))
        image = np.array(image)
        image = draw_bounding_boxes(image, boxes, confidences, class_ids)
        image = Image.fromarray(image)
        # resize back to original size
        image = image.resize(shape)

        # Display the result
        st.image(image, caption="Detected Objects", use_column_width=True)
        # clear memory and cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()