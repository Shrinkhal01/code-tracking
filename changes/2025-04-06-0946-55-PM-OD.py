import cv2

# Set a threshold value (minimum confidence) for object detection
thres = 0.50

# Capture video from the webcam
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)  # Try using the V4L2 backend for Linux systems
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Set video capture parameters (optional, you can adjust according to your needs)
cap.set(3, 1850)  # Width
cap.set(4, 1850)  # Height
cap.set(10, 70)   # Brightness

# Load the class names from coco.names
classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Paths to model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the pre-trained model
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Set model input properties
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start the video capture and perform object detection
while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success or img is None:
        print("Error: Failed to capture image.")
        break  # Exit the loop if frame capture fails

    # Perform object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Check if objects are detected
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Draw rectangle around detected objects and add class name and confidence score
... (truncated for brevity)