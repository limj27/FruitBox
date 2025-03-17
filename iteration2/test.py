import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("apple_classifier.h5")


# Load the image you want to test
image_path = 'apple_images/1/1.png'
apple_image = cv2.imread(image_path)

# Resize the image to the expected input size (e.g., 32x32 if that's your model input size)
apple_resized = cv2.resize(apple_image, (32, 32))

# Normalize the image for the model (if needed)
apple_resized = apple_resized / 255.0  # Assuming your model expects values between 0 and 1

# Add the batch dimension (model expects shape: (1, 32, 32, 3))
apple_resized = np.expand_dims(apple_resized, axis=0)

# Run the model prediction
prediction = model.predict(apple_resized)

# Get the predicted class (assuming it's a classification problem)
predicted_class = np.argmax(prediction)

# Print the prediction
print(f"Predicted class: {predicted_class}")
