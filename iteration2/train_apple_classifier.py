import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Define dataset path
DATASET_PATH = "apple_images/"  # Folder containing subfolders for each number (0-9)

# Load images and labels
X, y = [], []

for label in range(1, 10):  # Numbers from 0 to 9
    folder_path = os.path.join(DATASET_PATH, str(label))
    if not os.path.exists(folder_path):
        continue

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (32, 32)) / 255.0  # Resize to CNN input size
        X.append(image)
        y.append(label)

# Convert to numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")  # 10 output classes (0-9)
])

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("apple_classifier.h5")
print("Model saved as apple_classifier.h5")
