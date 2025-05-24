# object-detection-using-yolov8
**ABOUT THE PROJECT**
This project detects any object held in your hand in real time using your webcam. It combines MediaPipe for robust hand detection and YOLOv8 for object detection, allowing the system to display the name and holding time of any recognized object (e.g., bottle, book, cup, etc.) in your hand.
If your hand is empty, it displays a clear message.
If no hand is visible, it prompts you to show your hand.

**Technologies Used**
Python
OpenCV – for video capture and visualization
MediaPipe – for hand detection and tracking
YOLOv8 (Ultralytics) – for object detection

**How It Works**
MediaPipe detects your hand and provides its location in each frame.
YOLOv8 detects all objects in the frame.
The script checks which detected objects overlap with your hand (excluding "person").
If an object is detected in your hand, its name and hold time are displayed.
If your hand is empty: "No object detected in your hand."
If no hand is visible: "Show your hand to start detection."

**Customization**
To detect custom objects not in the COCO dataset, retrain YOLOv8 on your own dataset.
Adjust buffer size or detection thresholds in the script for smoother or faster response.

**Acknowledgements**
Ultralytics YOLOv8
Google MediaPipe
OpenCV
COCO Dataset
