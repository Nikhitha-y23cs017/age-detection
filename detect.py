import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import os

# Load model files
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

# Check if model files exist
for file in [face_proto, face_model, age_proto, age_model]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Required file not found: {file}")

# Load models
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

# Mean values and age categories
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, face_boxes

def predict_age(face_img, net):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    preds = net.forward()
    return age_list[preds[0].argmax()]

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
    frame, face_boxes = detect_faces(face_net, frame)
    print(f"Detected {len(face_boxes)} face(s)")
    for (x1, y1, x2, y2) in face_boxes:
        face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1),
                     max(0, x1-20):min(x2+20, frame.shape[1]-1)]
        try:
            age = predict_age(face, age_net)
        except Exception as e:
            age = "Unknown"
            print(f"Age prediction failed: {e}")
        cv2.putText(frame, f"Age: {age}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2_imshow(frame)
image_paths = ["girl1.jpg", "kid1.jpg", "man2.jpg"]
for path in image_paths:
    print(f"\nProcessing: {path}")
    process_image(path)
