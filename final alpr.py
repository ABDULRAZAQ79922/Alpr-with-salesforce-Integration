import cv2
import pytesseract
import re

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Set up the width and height for the frame
frameWidth = 1000
frameHeight = 480

# Load the Haar cascade for license plate detection
plateCascade = cv2.CascadeClassifier("C:/Users/abdul/Desktop/alpr/ANPR_with_opencv/haarcascade_russian_plate_number.xml")

def preprocess_image(img):
    """Convert image to grayscale, apply adaptive thresholding and denoising."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_plate_number(imgRoi):
    """Extract and process the license plate number from the ROI."""
    preprocessed_img = preprocess_image(imgRoi)
    text = pytesseract.image_to_string(preprocessed_img, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text = re.sub(r'\W+', '', text)
    return text.strip()

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates
    numberPlates = plateCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in numberPlates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        imgRoi = img[y:y + h, x:x + w]

        # Extract license plate number
        plate_number = extract_plate_number(imgRoi)
        if plate_number:
            print(f"Detected Number Plate: {plate_number}")
            cv2.putText(img, f"Detected Plate: {plate_number}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
