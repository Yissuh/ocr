import cv2
import numpy as np
from paddleocr import PaddleOCR

def main():
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Make a copy of the frame for drawing
        result = frame.copy()
        
        # Detect text
        results = ocr.ocr(frame, cls=True)
        
        # Process detection results
        if results[0]:
            for line in results[0]:
                # Get bounding box coordinates
                boxes = line[0]
                boxes = np.array(boxes).astype(np.int32).reshape((-1, 1, 2))
                
                # Draw bounding box
                cv2.polylines(result, [boxes], True, (0, 255, 0), 2)
                
                # Get detected text and confidence
                text = line[1][0]
                confidence = line[1][1]
                
                # Draw text above the bounding box
                text_position = (boxes[0][0][0], boxes[0][0][1] - 10)
                cv2.putText(result, f'{text} ({confidence:.2f})', 
                           text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Real-time Text Detection', result)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()