import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Define paths
main_dir = os.path.join('..', 'New Masks Dataset')
train_dir = os.path.join(main_dir, 'Train')
test_dir = os.path.join(main_dir, 'Test')
valid_dir = os.path.join(main_dir, 'Validation')

# Verify dataset existence
if not os.path.exists(main_dir):
    raise FileNotFoundError(f"Dataset directory '{main_dir}' not found.")

# Debug: Print directory contents
print("Main Dir Contents:", os.listdir(main_dir))
print("Train Dir Contents:", os.listdir(train_dir))

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    rotation_range=40,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

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
valid_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build model
model = Sequential([
    Conv2D(32, (3,3), padding='SAME', activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.5),
    Conv2D(32, (3,3), padding='SAME', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.5),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator
)

# Save model
os.makedirs(os.path.join('..', 'models'), exist_ok=True)
model.save(os.path.join('..', 'models', 'model.h5'))

# Plot results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.savefig(os.path.join('..', 'accuracy_plot.png'))
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.savefig(os.path.join('..', 'loss_plot.png'))
plt.show()

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator)
print('\nTest accuracy:', test_acc, '\tTest Loss:', test_loss)