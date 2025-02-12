import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# Load pre-trained EAST text detector model
net = cv2.dnn.readNet("model/frozen_east_text_detection.pb")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the output layer names for the EAST model
layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# Function to decode EAST output and extract bounding boxes
def decode_predictions(scores, geometry, conf_threshold=0.4):
    rects = []
    confidences = []
    
    height, width = scores.shape[2:4]

    for y in range(height):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(width):
            score = scoresData[x]
            if score < conf_threshold:
                continue

            # Extract geometry data
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)

            # Compute width & height
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute bounding box coordinates
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Store box coordinates and confidence scores
            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))

    return rects, confidences

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    H, W = frame.shape[:2]

    # Resize frame to meet EAST model requirements (multiples of 32)
    newW, newH = (320, 320)
    rW, rH = W / float(newW), H / float(newH)
    resized = cv2.resize(frame, (newW, newH))

    # Convert to blob for model input
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    scores, geometry = net.forward(layer_names)

    # Decode predictions
    rects, confidences = decode_predictions(scores, geometry)

    # Apply Non-Maximum Suppression (NMS)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Draw bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # Rescale to original frame size
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Add padding to boxes (optional)
        padding = 10
        startX, startY = max(0, startX - padding), max(0, startY - padding)
        endX, endY = min(W, endX + padding), min(H, endY + padding)

        # Draw bounding box
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Show the real-time text detection
    cv2.imshow("EAST Text Detection", orig)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
