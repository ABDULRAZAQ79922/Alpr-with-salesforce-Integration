import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model

# Set up Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "cfg/yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load pre-trained CNN model for character recognition
cnn_model = load_model('path_to_cnn_model.h5')

def detect_license_plate(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 is for license plates
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i[0]], confidences[i[0]]) for i in indices]

def recognize_text(img):
    img = cv2.resize(img, (128, 64))  # Resize to match CNN input
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    pred = cnn_model.predict(img)
    return ''.join([chr(c) for c in np.argmax(pred, axis=-1)])

def main():
    cap = cv2.VideoCapture(0)  # Capture from the webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_license_plate(frame)

        for (box, confidence) in detections:
            (x, y, w, h) = box
            plate_img = frame[y:y+h, x:x+w]
            text = recognize_text(plate_img)
            if not text:
                text = pytesseract.image_to_string(plate_img, config='--psm 8')
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('License Plate Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
