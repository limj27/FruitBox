import pyautogui
import time
from apple_detector import detect_apples
from utils import find_game_window

# Detect game window
monitor = find_game_window()
if not monitor:
    exit()

def find_valid_combinations(apple_grid):
    """Finds all adjacent apples that sum to 10."""
    valid_pairs = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    for (x, y), value in apple_grid.items():
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in apple_grid and apple_grid[neighbor] + value == 10:
                valid_pairs.append(((x, y), neighbor))

    return valid_pairs

def execute_drag(pair, grid_size=50):
    """Converts grid positions to screen coordinates and performs a drag action."""
    (x1, y1), (x2, y2) = pair

    # Convert grid coordinates to actual pixel positions
    start_x = monitor["left"] + x1 * grid_size + grid_size // 2
    start_y = monitor["top"] + y1 * grid_size + grid_size // 2
    end_x = monitor["left"] + x2 * grid_size + grid_size // 2
    end_y = monitor["top"] + y2 * grid_size + grid_size // 2

    # Perform drag
    pyautogui.moveTo(start_x, start_y, duration=0.1)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y, duration=0.1)
    pyautogui.mouseUp()

print(detect_apples(monitor))

# while True:
#     apple_grid = detect_apples(monitor)  # Get apple positions and values
#     pairs = find_valid_combinations(apple_grid)

#     if pairs:
#         for pair in pairs:
#             execute_drag(pair)  # Drag between apples that sum to 10

#     time.sleep(0.5)  # Delay to prevent excessive actions
