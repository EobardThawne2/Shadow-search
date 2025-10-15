"""
Bharatanatyam Mudra Classification Model Training Script

This script implements a deep learning pipeline for classifying Bharatanatyam mudras
using transfer learning with MobileNetV2.
"""

import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

class MudraClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32, learning_rate=0.001):
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def prepare_data(self, train_dir, val_dir):
        """Prepare data generators with augmentation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.num_classes = self.train_generator.num_classes
        self.class_names = list(self.train_generator.class_indices.keys())
        
        print(f"Found {self.train_generator.samples} training samples")
        print(f"Found {self.val_generator.samples} validation samples")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        
    def build_model(self):
        """Build the model using transfer learning"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom classification head
        inputs = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print("Model built successfully!")
        self.model.summary()
        
    def train(self, epochs=50, save_path='models/bharatanatyam_mudra_classifier.h5'):
        """Train the model"""
        if not self.model:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Training completed! Best model saved to {save_path}")
        
    def fine_tune(self, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        if not self.model:
            raise ValueError("Model not trained yet.")
            
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[1]  # MobileNetV2 is the second layer
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print("Starting fine-tuning...")
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            verbose=1
        )
        
        # Combine histories
        if self.history:
            for key in self.history.history:
                self.history.history[key].extend(fine_tune_history.history[key])
        else:
            self.history = fine_tune_history
            
    def plot_training_history(self):
        """Plot training and validation metrics"""
        if not self.history:
            print("No training history found.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-3 Accuracy
        axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Training Top-3 Acc')
        axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3 Acc')
        axes[1, 0].set_title('Model Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main training function"""
    # Paths
    train_dir = 'data/bharatanatyam_mudras/train'
    val_dir = 'data/bharatanatyam_mudras/validation'
    
    # Check if data directories exist
    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        print("Please create the dataset structure as described in README.md")
        return
        
    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        print("Please create the dataset structure as described in README.md")
        return
    
    # Initialize classifier
    classifier = MudraClassifier(img_size=(224, 224), batch_size=32, learning_rate=0.001)
    
    # Prepare data
    classifier.prepare_data(train_dir, val_dir)
    
    # Build model
    classifier.build_model()
    
    # Train model
    classifier.train(epochs=30)
    
    # Fine-tune (optional)
    # classifier.fine_tune(epochs=20)
    
    # Plot training history
    classifier.plot_training_history()
    
    print("Training completed successfully!")
    print("Model saved as 'models/bharatanatyam_mudra_classifier.h5'")


if __name__ == "__main__":
    main()