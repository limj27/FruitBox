import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import layers, models

# Define dataset path
DATASET_PATH = "apple_images/"  # Folder containing subfolders for each number (1-9)

# Load images and labels
X, y = [], []

for label in range(1, 10):  # Numbers from 1 to 9
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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Create an ImageDataGenerator instance with augmentations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator to the training data
datagen.fit(X_train)

# Load the pre-trained MobileNetV2 model, excluding the top (classification) layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model to prevent its weights from being updated during training
base_model.trainable = False

# Build the model with the pre-trained base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes (0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save("improved_apple_classifier.h5")
print("Model saved as improved_apple_classifier.h5")

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
