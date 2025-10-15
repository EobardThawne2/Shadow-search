"""
Simplified Mudra Classification Training Script

This script works with current TensorFlow versions and handles limited data.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_simple_model(num_classes):
    """Create a simple CNN model for mudra classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_simple_model():
    """Train a simple model with current data"""
    
    # Paths
    train_dir = 'data/bharatanatyam_mudras/train'
    val_dir = 'data/bharatanatyam_mudras/validation'
    
    print("Loading training data...")
    try:
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(224, 224),
            batch_size=4,  # Small batch for limited data
            validation_split=None,
            seed=123
        )
        
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(224, 224),
            batch_size=4,
            validation_split=None,
            seed=123
        )
        
        # Get class names
        class_names = train_dataset.class_names
        num_classes = len(class_names)
        
        print(f"Found {num_classes} classes: {class_names}")
        
        # Data augmentation for training
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        # Apply augmentation to training data
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        )
        
        # Optimize performance
        train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Create model
        print("Building model...")
        model = create_simple_model(num_classes)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'models/simple_mudra_classifier.keras',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                monitor='val_loss',
                restore_best_weights=True
            )
        ]
        
        # Train model
        print("Starting training...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            callbacks=callbacks,
            verbose=1
        )
        
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
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training completed!")
        print("Model saved as 'models/simple_mudra_classifier.keras'")
        print("Training history saved as 'training_history.png'")
        
        return model, history
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nPossible issues:")
        print("1. Make sure you have images in both train and validation directories")
        print("2. Each class should have at least 1 image in both train and validation")
        print("3. Check that image files are valid (jpg, png, etc.)")
        return None, None

def test_model(model_path='models/simple_mudra_classifier.keras'):
    """Test the trained model"""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    print("Loading model for testing...")
    model = tf.keras.models.load_model(model_path)
    
    # Test on validation data
    val_dir = 'data/bharatanatyam_mudras/validation'
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=4,
        shuffle=False
    )
    
    # Evaluate
    results = model.evaluate(val_dataset, verbose=1)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    
    # Make predictions
    predictions = model.predict(val_dataset)
    class_names = val_dataset.class_names
    
    print("\nPredictions on validation set:")
    for i, pred in enumerate(predictions[:5]):  # Show first 5 predictions
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred)
        print(f"Image {i+1}: {predicted_class} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    print("=" * 60)
    print("BHARATANATYAM MUDRA CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Train model
    model, history = train_simple_model()
    
    if model is not None:
        print("\n" + "=" * 60)
        print("TESTING TRAINED MODEL")
        print("=" * 60)
        
        # Test model
        test_model()
        
        print("\n✅ Training and testing completed!")
    else:
        print("\n❌ Training failed. Please check your data and try again.")