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
Yolov5x which is the largest version of the model was used for training and evaluation from scratch. (No pretrained weights were used)
- The training set consisted of 8185 images.
- The validation set consisted of 584 images.
- The test set consisted of 2920 images.

The image size was set to 640x640 and the batch size was set to 2 due to the data being large and avoiding GPUs running out of memory.
The training was done on 2 x 3060 Ti Nvidia GPUs each consisting of 8gb vram for approximately 3 hours.

### Graphs for training and validation for various metrics.
![Graphs](/assets/results.png)

### Confusion Matrix for the validation set.
![Confusion Matrix](/assets/confusion_matrix.png)

## Quantization and its benefits
Quantization is a technique to reduce the size of the model by reducing the precision of the weights and activations.
Quantization involves converting the weights of the model from 32-bit floating-point precision to lower-precision formats, such as 8-bit integers, while minimizing the loss of accuracy.

Some benefits of quantizing yolov5 model are:-
- Reduced Memory Requirements: Quantizing YOLOv5 can significantly reduce the memory requirements of the model, making it more suitable for deployment on mobile devices with limited memory resources. This can also help to reduce the power consumption and increase the battery life of the device.
- Improved Computational Efficiency: Quantized YOLOv5 models can be executed more efficiently on mobile devices, due to the reduced precision of the weights. This can lead to faster inference times and lower latency, which is particularly important for real-time applications such as object detection in mobile cameras.
- Reduced Model Size: Quantization can also help to reduce the size of the YOLOv5 model, making it easier to deploy and distribute to mobile devices. This can also help to reduce the network bandwidth requirements and storage costs associated with the deployment of the model.
- Compatibility with Mobile Hardware: Quantized YOLOv5 models can be optimized for mobile hardware architectures. This can further improve the performance and efficiency of the model on mobile devices.

## Results
- We were able to achieve exceptional results with the Yolov5 model on the SKU110K dataset
- But remember the main objective of this project was to quantize the model and deploy it on edge devices for real-time detection in dense environments.
- The model was quantized into various formats and different precisions and the results were compared to one another on test set.

| Index | Model                         | Size (mb) | Precision | Recall | mAP50 | F1 Score |
| ------| ----------------------------- | --------- | --------- | ------ | ----- | -------- |
| 1     | Pytorch (.pt)                 | 164       | 0.930     | 0.868  | 0.922 | 0.8979   |
| 2     | Tensorflow (.pb)              | 329       | 0.925     | 0.868  | 0.919 | 0.8955   |
| 3     | Tflite Float 32 (.tflite)     | 328       | 0.925     | 0.868  | 0.919 | 0.8955   |
| 4     | Tflite Float 16 (.tflite)     | 164       | 0.925     | 0.868  | 0.919 | 0.8955   |
| 5     | Tflite Int 8 (.tflite)        | 83        | 0.917     | 0.865  | 0.915 | 0.8902   |

> **_NOTE:_** Individual results and plots for each model can be found in the repository in test folder.

Theoritically the model should have been quantized to INT8 precision with minimal loss in accuracy and we could see that the model was quantized to INT8 precision with a loss in accuracy of 0.007 which is very minimal and works in our favor.

But, when these models were inferenced and compared to one another the int 8 model performs the best in terms of accuracy.
Amazing right? :)

## Inference
- As we can see the models in the middle perform very similar to one another.
- Here are some inference results for various models.


<p float="left">
  <img src="/assets/pt_img.png" width="412" />
  <img src="/assets/tflite_fp32.png" width="412" /> 
</p>
                    Pytorch (.pt)                   Tflite Float 32 (.tflite)

<p float="left">
  <img src="/assets/tflite_fp16.png" width="412" />
  <img src="/assets/tflite_int8.png" width="412" />
</p>
                    Tflite Float 16 (.tflite)                   Tflite Int 8 (.tflite)

## Replicating the Work
The model weights are not being shared but you can mimic the whole pipeline using the notebook provided in the repository to obtain them.

## Advantages of using SKU110K dataset on Yolov5 model
- Large and Diverse: The dataset contains a large and diverse set of product images, covering a wide range of categories, brands, and styles. This diversity can help the YOLOv5 model learn to recognize a variety of products and adapt to different retail store settings.
- Realistic Scenarios: The images in the dataset are captured in real-world retail store settings, with varying illumination, background clutter, and occlusion. This can help the YOLOv5 model learn to recognize products in realistic scenarios, which can be particularly useful in retail store settings.
- Annotation Quality: The dataset includes high-quality annotations for each product image, including bounding box coordinates and class labels. This can help the YOLOv5 model learn to accurately detect and classify products, which is critical for many retail store applications.
- Faster Inference: YOLOv5 is known for its speed and efficiency, making it a good choice for real-time object detection applications in retail stores. The model can quickly process large volumes of product images and provide accurate object detection results.

## Limitations of using SKU110K dataset on Yolov5 model
- Limited Domain: While the dataset contains a large and diverse set of product images, it is still limited to the retail domain. This means that the YOLOv5 model trained on this dataset may not perform well on other object detection tasks outside of retail stores.
- Limited Annotations: While the dataset includes high-quality annotations for each product image, the annotations are limited to bounding box coordinates and class labels. This means that the YOLOv5 model trained on this dataset may not be able to recognize finer details of the products, such as specific features or attributes.
- Limited Generalizability: While the dataset contains a diverse set of product images, the images are still limited to specific retail store settings and conditions. This means that the YOLOv5 model trained on this dataset may not generalize well to other retail store settings or to different types of product images.

## Real World Implementations
- Inventory Management: Retail stores can use object detection models to automatically track and manage inventory levels for different products. By analyzing product images captured in real-time, the YOLOv5 model can detect and count the number of products on shelves or in storage, and alert store managers when inventory levels are running low.
- Loss Prevention: Object detection models can also be used to identify and prevent theft or fraud in retail stores. By analyzing surveillance camera footage or product images, the YOLOv5 model can detect and track suspicious behavior, such as shoplifting or product tampering, and alert store security personnel.
- Customer Behavior Analysis: Retail stores can use object detection models to analyze customer behavior and preferences. By analyzing product images and tracking customer movements, the YOLOv5 model can identify popular product categories and display placements, and provide insights into customer shopping behavior.
- Product Recommendation: Object detection models can also be used to provide personalized product recommendations to customers. By analyzing customer images or product images captured in real-time, the YOLOv5 model can identify the products that a customer is interested in and recommend complementary or similar products.