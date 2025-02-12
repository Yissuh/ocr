import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological transformations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Use MSER to detect text-like regions
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(morph)

    # Filter and draw bounding boxes
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        
        # Calculate aspect ratio and area
        aspect_ratio = w / float(h)
        area = w * h
        
        # Apply refined size filtering for real-time text detection
        if 0.3 < aspect_ratio < 5 and 500 < area < 80000 and h > 12:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Text Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
