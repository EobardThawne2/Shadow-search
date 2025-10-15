"""
Simple Binary Classification Demo - Pataka vs Not-Pataka

This is a simplified version for demo purposes with limited data.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_demo_data():
    """Create dummy 'non-pataka' images for binary classification demo"""
    # Create a simple non-pataka class with random images for demo
    non_pataka_dir = 'data/bharatanatyam_mudras/train/non_pataka'
    os.makedirs(non_pataka_dir, exist_ok=True)
    
    # Create some random noise images as placeholder
    for i in range(5):
        # Create random image
        random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(random_img)
        img.save(os.path.join(non_pataka_dir, f'random_{i}.jpg'))
    
    # Validation
    val_non_pataka_dir = 'data/bharatanatyam_mudras/validation/non_pataka'
    os.makedirs(val_non_pataka_dir, exist_ok=True)
    for i in range(2):
        random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(random_img)
        img.save(os.path.join(val_non_pataka_dir, f'random_{i}.jpg'))
    
    print("Created demo non-pataka images for binary classification")

def train_demo_model():
    """Train a simple demo model"""
    # Create demo data
    create_demo_data()
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'data/bharatanatyam_mudras/train',
        target_size=(224, 224),
        batch_size=4,  # Small batch size due to limited data
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        'data/bharatanatyam_mudras/validation',
        target_size=(224, 224),
        batch_size=4,
        class_mode='categorical'
    )
    
    # Build simple model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Training with {train_generator.num_classes} classes")
    print(f"Class names: {list(train_generator.class_indices.keys())}")
    
    # Train
    history = model.fit(
        train_generator,
        epochs=10,  # Short training for demo
        validation_data=val_generator,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/demo_mudra_classifier.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('demo_training_history.png')
    plt.show()
    
    print("Demo training completed!")
    print("Model saved as 'models/demo_mudra_classifier.h5'")

if __name__ == "__main__":
    train_demo_model()