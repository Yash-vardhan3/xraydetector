import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import cv2
import matplotlib.pyplot as plt
import pickle

# Define paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Define ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Save the trained model in TensorFlow's native format
model.save('healthcare_diagnostics_model.h5')

# Save the model using pickle
with open('healthcare_diagnostics_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predict function
def predict_image(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read the image file {image_path}. Please check the file path or the file integrity.")
    
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'

# Test the function
test_image_path = 'dataset/test/normal/IM-0016-0001.jpeg'
try:
    prediction = predict_image(test_image_path, model)
    print(f'The prediction for the test image is: {prediction}')
except Exception as e:
    print(e)
