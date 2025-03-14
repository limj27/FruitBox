import pyautogui

# Locate the game window using a reference image
game_location = pyautogui.locateOnScreen("fruitbox_reference.png", confidence=0.8)

if game_location:
    monitor = {
        "top": game_location.top,
        "left": game_location.left,
        "width": game_location.width,
        "height": game_location.height
    }
    print(f"Detected game area: {monitor}")
else:
    print("Game window not found. Make sure it's visible on any monitor.")
