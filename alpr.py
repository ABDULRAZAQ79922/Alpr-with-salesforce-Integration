import pytesseract


# Set the path to the Tesseract executable

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe' 
import cv2

# Set up the width and height for the frame
frameWidth = 1000
frameHeight = 480

# Load the Haar cascade for license plate detection
plateCascade = cv2.CascadeClassifier("C:/Users/abdul/Desktop/alpr/ANPR_with_opencv/haarcascade_russian_plate_number.xml")
minArea = 500

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("Number Plate", imgRoi)

            # Use Tesseract to extract the license plate number
            plate_number = pytesseract.image_to_string(imgRoi, config='--psm 8')
            print(f"Detected Number Plate: {plate_number.strip()}")

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"C:/Users/abdul/Desktop/alpr/ANPR_with_opencv/IMAGES{str(count)}.jpg", imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1

cap.release()
cv2.destroyAllWindows()
