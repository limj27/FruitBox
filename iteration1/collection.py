from pynput import mouse
import json
import time
import pyautogui
import cv2
import mss
import numpy as np

dataset = []

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

recording = False
drag_start = None
drag_end = None

def on_click(x, y, button, pressed):
    global recording, drag_start, drag_end

    if pressed and button == mouse.Button.left:
        drag_start = (x, y)  # Record start point
        drag_end = None  # Reset end point
        recording = True

    elif not pressed and button == mouse.Button.left:
        recording = False
        if drag_start and drag_end:
            frame = capture_screen()
            dataset.append({"image": frame.tolist(), "drag_path": [drag_start, drag_end]})
            print(f"Saved drag path: {drag_start} -> {drag_end}")
        drag_start = None  # Reset start point after the drag

def on_move(x, y):
    global drag_start, drag_end
    if drag_start:
        drag_end = (x, y)  # Continuously update end point as the mouse moves

# Start listening for mouse events system-wide
mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
mouse_listener.start()

print("Recording... Drag to collect data. Press 'q' in the terminal to stop.")

try:
    while True:
        if input().lower() == 'q':  # Quit and save dataset
            with open("fruitbox_data.json", "w") as f:
                json.dump(dataset, f)
            print("Dataset saved!")
            break
except KeyboardInterrupt:
    pass

mouse_listener.stop()
