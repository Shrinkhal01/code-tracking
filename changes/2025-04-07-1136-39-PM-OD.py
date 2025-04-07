import cv2
import os

thres = 0.5  # Confidence threshold

# Load class names
with open("coco.names", "rt") as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Use video file instead of webcam
cap = cv2.VideoCapture("video2.mp4")

# Output folder for processed frames
os.makedirs("output", exist_ok=True)

frame_count = 0
print("Processing video...")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Video ended or failed to read frame.")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 <= classId - 1 < len(classNames):
                label = f"{classNames[classId - 1]}: {round(confidence * 100, 2)}%"
                cv2.rectangle(img, box, (0, 255, 0), 2)
                cv2.putText(img, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save every frame with detections
    cv2.imwrite(f"output/frame_{frame_count:04d}.jpg", img)
    frame_count += 1

cap.release()
print("Done! Processed frames are in the 'output/' directory.")