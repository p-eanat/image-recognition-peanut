# Testing and applications



import os
import tensorflow as tf
from tensorflow import keras
import numpy as np



# Load model
model = tf.keras.models.load_model("model.h5")

# Check model structure
print(model.summary())

# Test model

img_path = ""
class_names = [] # Fill in class names


img_height = img_width = 150
img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)