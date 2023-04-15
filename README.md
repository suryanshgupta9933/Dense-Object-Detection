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

## Training and Evaluation
The training was done on SKU110K dataset for a maximum of 50 epochs but during the training we could see the metrics saturate around 30 epochs when the mAP50 was almost 0.6 on the training set.
- The training set consisted of 8185 images.
- The validation set consisted of 584 images.
- The test set consisted of 2920 images.

The image size was set to 640x640 and the batch size was set to 2 due to the data being large and avoiding GPUs running out of memory.
The training was done on 2 x 3060 Ti Nvidia GPUs each consisting of 8gb vram for approximately 3 hours.

<img src="assets/train__batch0.jpg" width="128"/>

## Advantages
- Large and Diverse: The dataset contains a large and diverse set of product images, covering a wide range of categories, brands, and styles. This diversity can help the YOLOv5 model learn to recognize a variety of products and adapt to different retail store settings.
- Realistic Scenarios: The images in the dataset are captured in real-world retail store settings, with varying illumination, background clutter, and occlusion. This can help the YOLOv5 model learn to recognize products in realistic scenarios, which can be particularly useful in retail store settings.
- Annotation Quality: The dataset includes high-quality annotations for each product image, including bounding box coordinates and class labels. This can help the YOLOv5 model learn to accurately detect and classify products, which is critical for many retail store applications.
- Faster Inference: YOLOv5 is known for its speed and efficiency, making it a good choice for real-time object detection applications in retail stores. The model can quickly process large volumes of product images and provide accurate object detection results.

## Limitations
- Limited Domain: While the dataset contains a large and diverse set of product images, it is still limited to the retail domain. This means that the YOLOv5 model trained on this dataset may not perform well on other object detection tasks outside of retail stores.
- Limited Annotations: While the dataset includes high-quality annotations for each product image, the annotations are limited to bounding box coordinates and class labels. This means that the YOLOv5 model trained on this dataset may not be able to recognize finer details of the products, such as specific features or attributes.
- Limited Generalizability: While the dataset contains a diverse set of product images, the images are still limited to specific retail store settings and conditions. This means that the YOLOv5 model trained on this dataset may not generalize well to other retail store settings or to different types of product images.

## Real World Implementation
- Inventory Management: Retail stores can use object detection models to automatically track and manage inventory levels for different products. By analyzing product images captured in real-time, the YOLOv5 model can detect and count the number of products on shelves or in storage, and alert store managers when inventory levels are running low.
- Loss Prevention: Object detection models can also be used to identify and prevent theft or fraud in retail stores. By analyzing surveillance camera footage or product images, the YOLOv5 model can detect and track suspicious behavior, such as shoplifting or product tampering, and alert store security personnel.
- Customer Behavior Analysis: Retail stores can use object detection models to analyze customer behavior and preferences. By analyzing product images and tracking customer movements, the YOLOv5 model can identify popular product categories and display placements, and provide insights into customer shopping behavior.
- Product Recommendation: Object detection models can also be used to provide personalized product recommendations to customers. By analyzing customer images or product images captured in real-time, the YOLOv5 model can identify the products that a customer is interested in and recommend complementary or similar products.

## Quantization and its benefits
- Quantization is a technique to reduce the size of the model by reducing the precision of the weights and activations.
- Quantization involves converting the weights of the model from 32-bit floating-point precision to lower-precision formats, such as 8-bit integers, while minimizing the loss of accuracy.

Some benefits of quantizing yolov5 model are:-
- Reduced Memory Requirements: Quantizing YOLOv5 can significantly reduce the memory requirements of the model, making it more suitable for deployment on mobile devices with limited memory resources. This can also help to reduce the power consumption and increase the battery life of the device.
- Improved Computational Efficiency: Quantized YOLOv5 models can be executed more efficiently on mobile devices, due to the reduced precision of the weights. This can lead to faster inference times and lower latency, which is particularly important for real-time applications such as object detection in mobile cameras.
- Reduced Model Size: Quantization can also help to reduce the size of the YOLOv5 model, making it easier to deploy and distribute to mobile devices. This can also help to reduce the network bandwidth requirements and storage costs associated with the deployment of the model.
- Compatibility with Mobile Hardware: Quantized YOLOv5 models can be optimized for mobile hardware architectures. This can further improve the performance and efficiency of the model on mobile devices.