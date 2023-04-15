# Yolov5-SKU110K
Object Detection Model using Yolov5 by Ultralytics on SKU110K dataset and Post Quantization.

## The Yolov5 model by Ultralytics is an efficient object detection model that can be quantized and deployed on edge devices for real-time detection in dense environments.
Some of the key features of Yolov5 are:
- It is fast (30+ FPS) and accurate.
- It has small model size (from 3.2MB to 253MB) which makes it suitable for deployment on edge devices.
- It provides several versions (yolov5s, yolov5m, yolov5l, yolov5x) for different accuracy-speed tradeoffs.
- It has an active development community which provides frequent updates.
- It has PyTorch, TensorFlow and ONNX versions which provides flexibility in the choice of framework.
- It can be quantized to INT8 precision with minimal loss in accuracy making it suitable for edge devices with low memory and compute.

## Post Training Quantization
This does not require any modifications to the network, so you can convert a previously trained network into a quantized model.
This can be done in various different ways:-
- If no quantization is performed, which simply means a conversion of tf model to tflite model.
- If only weights of the model are quantized also called hybrid quantization.
- If both weights and activation are quantized also called full quantization.

## Training Pipeline
The training was done on SKU110K dataset for a maximum of 50 epochs but during the training we could see the metrics saturate around 30 epochs when the mAP50 was almost 0.6 on the training set.
- The training set consisted of 8185 images.
- The validation set consisted of 584 images.
- The test set consisted of 2920 images.
The image size was set to 640x640 and the batch size was set to 2 due to the data being large and avoiding GPUs running out of memory.
The training was done on 2 x 3060 Ti Nvidia GPUs each consisting of 8gb vram for approximately 3 hours.
