import tensorflow as tf
import numpy as np
import json
import cv2


# Load dataset
with open("fruitbox_data.json", "r") as f:
    dataset = json.load(f)

# Preprocess images and drag paths
X = []
y = []

for data in dataset:
    image = np.array(data["image"], dtype=np.uint8)
    image = cv2.resize(image, (128, 128)) / 255.0  # Normalize
    X.append(image)

    # Normalize drag path (scaled to the range 0-1)
    drag_path = np.array(data["drag_path"], dtype=np.float32)

    # Ensure drag_path has exactly 2 points, each with (x, y) coordinates
    if len(drag_path) < 2:
        # If there are fewer than 2 points, pad with zeros (or an appropriate value)
        drag_path = np.pad(drag_path, ((0, 2 - len(drag_path)), (0, 0)), 'constant')
    else:
        # If there are more than 2 points, crop to the first 2 points
        drag_path = drag_path[:2]

    drag_path = drag_path / 1920  # Normalize by screen width
    y.append(drag_path)

X = np.array(X, dtype=np.float32)

# Ensure y is a NumPy array
y = np.array(y, dtype=np.float32)  
print("Shape of y:", y.shape)  # Debug print

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(4, activation="sigmoid"),  # 4 outputs for 2 (x, y) pairs
    tf.keras.layers.Reshape((2, 2))  # Reshape output to (2, 2) to match y
])

model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

# Train the model
model.fit(X, y, epochs=10, batch_size=8)

# Save the trained model
model.save("fruitbox_ai.h5")
