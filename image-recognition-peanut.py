"""
Description: Image recognition using machine learning (for UAS?)
Sources: Based on Google's image classification course https://developers.google.com/machine-learning/practica/image-classification
Author: Amy
Date created: Sept. 3, 2022

"""



import os
# There's a bug in tensorflow, so these may have squiggly lines :(
# (Don't worry, the program still runs fine)
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Import weights from pre-trained Inception v3
weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None) # We don't include the classification layers (a.k.a. top), since we don't need them for feature extraction
pre_trained_model.load_weights(weights_file)

# Set the pretrained model to be non-trainable for now
for layer in pre_trained_model.layers:
  layer.trainable = False

# We will use the layer called "mixed7" which is a 7x7 feature map 
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# Add a fully connected classifier:
# 1. Flatten the output layer to 1 dimension
our_input = layers.Flatten()(last_output)
# 2. Add a fully connected layer with 1,024 hidden units and ReLU activation
our_input = layers.Dense(1024, activation='relu')(our_input)
# 3. Add a dropout rate of 0.2
our_inputx = layers.Dropout(0.2)(our_input)
# 4. Add a final sigmoid layer for classification
our_input = layers.Dense(1, activation='sigmoid')(our_input)

# Configure and compile the model
model = Model(pre_trained_model.input, our_input)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['acc']) # Metric: accuracy



# Define our example directories and files
train_dir = os.path("/photos/train")
validation_dir = os.path("/photos/validation")

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,                  # This is the source directory for training images
    target_size=(150, 150),     # All images will be resized to 150x150
    batch_size=20,              # Batch size 20
    class_mode='binary'         # Since we use binary_crossentropy loss, we need binary labels
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)



# Train the model!
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

# Unfreeze all models after "mixed6"
unfreeze = False
for layer in pre_trained_model.layers:
  if unfreeze:
    layer.trainable = True
  if layer.name == 'mixed6':
    unfreeze = True

# As an optimizer, here we will use SGD with a very low learning rate (0.00001)
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.00001, momentum=0.9), metrics=['acc'])

# Train again!
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)



# Graph some results

# Retrieve a list of accuracy results on training and validation datasets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation datasets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

# Save model
model.save("model.h5")