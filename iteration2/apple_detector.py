import cv2
import numpy as np
import tensorflow as tf
import mss
import matplotlib.pyplot as plt

# Load trained CNN model for apple detection
model = tf.keras.models.load_model("apple_classifier.h5")

def capture_screen(monitor):
    """Capture the game screen and return an image."""
    with mss.mss() as sct:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    return frame

import cv2
import numpy as np

def detect_apples(monitor, grid_size=50):
    """Detect apples and their numbers, returning a structured grid."""
    frame = capture_screen(monitor)  # Capture the screen (or replace with your frame)
    height, width, _ = frame.shape

    apple_grid = {}  # Dictionary to store apples {(grid_x, grid_y): number}

    # Convert the image to grayscale for easier contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Show the grayscale image for debugging
    cv2.imshow('Grayscale Image', gray)

    # Try using adaptive thresholding for better segmentation of the apples
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Show the thresholded image for debugging
    cv2.imshow('Thresholded Image', thresh)

    # Find contours in the thresholded image (this will detect possible apples)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours for visualization
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)  # Draw contours in green

    # Show the contours drawn on the frame (for debugging purposes)
    cv2.imshow('Contours', contour_frame)

    # Loop through the contours to find and classify apples
    for idx, contour in enumerate(contours):
        # Get bounding box around each detected apple
        x, y, w, h = cv2.boundingRect(contour)
        cv2.imshow(f"Contour {idx}", contour)
        # Filter out small contours that are unlikely to be apples
        if w < 10 or h < 10:  # Adjust these thresholds as needed
            continue

        # Optional: You can check the contour area to filter out smaller or irrelevant contours
        if cv2.contourArea(contour) < 100:  # Adjust this value if needed
            continue

        # Crop the detected apple region from the original image
        apple_crop = frame[y:y+h, x:x+w]
        
        # Resize the cropped apple to fit the model input (assuming model expects 32x32)
        apple_resized = cv2.resize(apple_crop, (32, 32))
        
        # Normalize and add batch dimension to the image for model input
        apple_resized = np.expand_dims(apple_resized, axis=0)  # Shape: (1, 32, 32, 3)
        apple_resized = apple_resized / 255.0  # Normalize to [0, 1]
        
        # Predict the class of the apple (the number on the apple)
        prediction = model.predict(apple_resized)
        apple_number = np.argmax(prediction)  # Get the predicted class

        # Only store if it's a valid number
        if apple_number > 0:
            grid_x = x // grid_size
            grid_y = y // grid_size
            apple_grid[(grid_x, grid_y)] = apple_number

    # Wait until any key is pressed to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return apple_grid
