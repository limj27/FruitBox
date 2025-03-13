import pyautogui
import tensorflow as tf
import numpy as np
import cv2
import mss
import time

# Load trained model
model = tf.keras.models.load_model("fruitbox_ai.h5", custom_objects={"MeanSquaredError": tf.keras.losses.MeanSquaredError})

def find_game_window():
    game_location = pyautogui.locateOnScreen("fruitbox_reference.png", confidence=0.8)
    if game_location:
        monitor = {
            "top": game_location.top,
            "left": game_location.left,
            "width": game_location.width,
            "height": game_location.height
        }
        print(f"Detected game area: {monitor}")
        return monitor
    else:
        print("Game window not found. Make sure it's visible on any monitor.")
        return None

monitor = find_game_window()
if not monitor:
    exit()

def capture_screen():
    with mss.mss() as sct:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return frame

while True:
    # Capture screen frame
    frame = capture_screen()

    # Preprocess image: resize and normalize
    image = cv2.resize(frame, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict drag path (relative to game window size)
    prediction = model.predict(image)[0]
    
    # Scale predictions to fit within the detected game window
    x_scaled = prediction[:, 0] * monitor["width"] + monitor["left"]
    y_scaled = prediction[:, 1] * monitor["height"] + monitor["top"]
    path = list(zip(x_scaled.astype(int), y_scaled.astype(int)))
    
    print("Extracted path (scaled):", path)

    # Ensure coordinates stay within the game window bounds
    path = [
        (min(max(x, monitor["left"]), monitor["left"] + monitor["width"] - 1),
         min(max(y, monitor["top"]), monitor["top"] + monitor["height"] - 1))
        for x, y in path
    ]

    # Perform the drag motion if path has more than 1 point
    if len(path) > 1:
        pyautogui.moveTo(path[0][0], path[0][1], duration=0.1)
        pyautogui.mouseDown()

        for x, y in path[1:]:
            pyautogui.moveTo(x, y, duration=0.05)

        pyautogui.mouseUp()
    
    time.sleep(0.5)
