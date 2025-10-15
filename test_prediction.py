"""
Simple Prediction Script for Trained Mudra Model
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def predict_mudra(image_path, model_path='models/simple_mudra_classifier.keras'):
    """Predict mudra from an image"""
    
    # Load the trained model
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    
    # Class names (these should match your training data)
    class_names = ['arala', 'ardhachandra', 'ardhapataka', 'kartarimukha', 'mayura', 
                   'mushtika', 'pataka', 'shukatunda', 'sikhara', 'tripataka']
    
    # Load and preprocess the image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    
    # Get the predicted class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Display results
    plt.figure(figsize=(10, 6))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Input Image')
    plt.axis('off')
    
    # Show prediction results
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, predictions[0])
    plt.yticks(y_pos, class_names)
    plt.xlabel('Confidence')
    plt.title('Prediction Results')
    
    # Highlight the predicted class
    max_idx = np.argmax(predictions[0])
    plt.barh(max_idx, predictions[0][max_idx], color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 PREDICTION RESULT:")
    print(f"   Predicted Mudra: {predicted_class.upper()}")
    print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    
    return predicted_class, confidence

def test_with_training_images():
    """Test the model with some training images"""
    import os
    
    # Test with a pataka image
    pataka_dir = 'data/bharatanatyam_mudras/train/pataka'
    if os.path.exists(pataka_dir):
        image_files = [f for f in os.listdir(pataka_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = os.path.join(pataka_dir, image_files[0])
            print(f"Testing with training image: {test_image}")
            predict_mudra(test_image)
        else:
            print("No image files found in pataka directory")
    else:
        print("Pataka directory not found")

if __name__ == "__main__":
    print("=" * 60)
    print("MUDRA PREDICTION TEST")
    print("=" * 60)
    
    # Test with a training image
    test_with_training_images()
    
    print("\n" + "=" * 60)
    print("To test with your own image, use:")
    print("predict_mudra('path/to/your/image.jpg')")
    print("=" * 60)