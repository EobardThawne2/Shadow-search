"""
Utility functions for Bharatanatyam Mudra Classification Project
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from config import MUDRA_CLASSES, COLORS, MUDRA_DESCRIPTIONS

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for model prediction"""
    try:
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
            
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def create_class_weight(train_generator):
    """Create class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get class counts
    class_counts = {}
    for class_name in train_generator.class_indices:
        class_dir = os.path.join(train_generator.directory, class_name)
        count = len([f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        class_counts[class_name] = count
    
    # Compute class weights
    classes = list(range(len(MUDRA_CLASSES)))
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=[train_generator.class_indices[name] for name in train_generator.class_indices]
    )
    
    return dict(zip(classes, class_weights))

def plot_sample_images(data_generator, num_samples=10, save_path=None):
    """Plot sample images from data generator"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    # Get a batch of images
    batch_images, batch_labels = next(data_generator)
    
    for i in range(min(num_samples, len(batch_images))):
        # Get image and label
        img = batch_images[i]
        label_idx = np.argmax(batch_labels[i])
        label_name = MUDRA_CLASSES[label_idx]
        
        # Plot image
        axes[i].imshow(img)
        axes[i].set_title(f'{label_name}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_model_info(model, save_path):
    """Save model architecture and configuration info"""
    model_info = {
        'model_name': 'Bharatanatyam Mudra Classifier',
        'architecture': 'MobileNetV2 + Custom Head',
        'total_params': model.count_params(),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'num_classes': len(MUDRA_CLASSES),
        'class_names': MUDRA_CLASSES
    }
    
    with open(save_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model info saved to {save_path}")

def calculate_model_size(model_path):
    """Calculate model size in MB"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

def create_prediction_report(predictions, true_labels, class_names, save_path=None):
    """Create a detailed prediction report"""
    from sklearn.metrics import classification_report, accuracy_score
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Generate classification report
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=class_names,
        output_dict=True
    )
    
    # Create summary
    summary = {
        'overall_accuracy': accuracy,
        'total_samples': len(predictions),
        'num_classes': len(class_names),
        'per_class_accuracy': {},
        'model_performance': 'Excellent' if accuracy > 0.95 else 
                           'Good' if accuracy > 0.85 else
                           'Acceptable' if accuracy > 0.75 else 'Poor'
    }
    
    # Add per-class metrics
    for class_name in class_names:
        if class_name in report:
            summary['per_class_accuracy'][class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1_score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Prediction report saved to {save_path}")
    
    return summary

def visualize_mudra_info(mudra_name):
    """Display information about a specific mudra"""
    if mudra_name.lower() in MUDRA_DESCRIPTIONS:
        info = MUDRA_DESCRIPTIONS[mudra_name.lower()]
        print(f"\n{'='*50}")
        print(f"MUDRA: {info['name'].upper()}")
        print(f"{'='*50}")
        print(f"Description: {info['description']}")
        print(f"Usage: {info['usage']}")
        print(f"{'='*50}\n")
    else:
        print(f"Information not available for mudra: {mudra_name}")

def create_training_summary(history, model_path, save_path=None):
    """Create a summary of training process"""
    final_epoch = len(history.history['accuracy']) - 1
    
    summary = {
        'training_summary': {
            'total_epochs': final_epoch + 1,
            'final_train_accuracy': float(history.history['accuracy'][final_epoch]),
            'final_val_accuracy': float(history.history['val_accuracy'][final_epoch]),
            'final_train_loss': float(history.history['loss'][final_epoch]),
            'final_val_loss': float(history.history['val_loss'][final_epoch]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_accuracy_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
            'model_size_mb': calculate_model_size(model_path)
        }
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved to {save_path}")
    
    return summary

def preprocess_for_prediction(image_input, target_size=(224, 224)):
    """Preprocess image for prediction - supports various input types"""
    if isinstance(image_input, str):
        # File path
        img = cv2.imread(image_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        # NumPy array (from OpenCV or similar)
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            img = image_input
        else:
            raise ValueError("Invalid image array shape")
    else:
        raise ValueError("Unsupported image input type")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def draw_hand_landmarks(image, landmarks, connections=None):
    """Draw hand landmarks on image (placeholder for MediaPipe integration)"""
    # This is a placeholder function for future MediaPipe integration
    # You can expand this to use MediaPipe for hand landmark detection
    return image

def calculate_gesture_similarity(gesture1_features, gesture2_features):
    """Calculate similarity between two gesture feature vectors"""
    # Cosine similarity
    dot_product = np.dot(gesture1_features, gesture2_features)
    norm_1 = np.linalg.norm(gesture1_features)
    norm_2 = np.linalg.norm(gesture2_features)
    
    similarity = dot_product / (norm_1 * norm_2)
    return similarity

def create_mudra_reference_chart(save_path='mudra_reference_chart.png'):
    """Create a reference chart for all mudras"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, mudra_name in enumerate(MUDRA_CLASSES):
        ax = axes[i]
        
        # Create a placeholder image (you can replace with actual mudra images)
        ax.text(0.5, 0.7, mudra_name.upper(), ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        if mudra_name in MUDRA_DESCRIPTIONS:
            description = MUDRA_DESCRIPTIONS[mudra_name]['description']
            # Wrap text
            wrapped_text = '\n'.join([description[j:j+30] for j in range(0, len(description), 30)])
            ax.text(0.5, 0.3, wrapped_text, ha='center', va='center', 
                   fontsize=8, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
            spine.set_visible(True)
    
    plt.suptitle('Bharatanatyam Mudra Reference Chart', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Mudra reference chart saved to {save_path}")

def validate_image_format(image_path):
    """Validate if image format is supported"""
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_ext = os.path.splitext(image_path)[1].lower()
    return file_ext in supported_formats

def batch_predict_images(model, image_directory, batch_size=32):
    """Predict mudras for all images in a directory"""
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(image_directory, ext)))
    
    if not image_paths:
        print(f"No images found in {image_directory}")
        return []
    
    predictions = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = preprocess_for_prediction(path)
            if img is not None:
                batch_images.append(img[0])  # Remove batch dimension
        
        if batch_images:
            batch_array = np.array(batch_images)
            batch_predictions = model.predict(batch_array)
            
            for j, pred in enumerate(batch_predictions):
                predicted_class = MUDRA_CLASSES[np.argmax(pred)]
                confidence = np.max(pred)
                predictions.append({
                    'image_path': batch_paths[j],
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
    
    return predictions