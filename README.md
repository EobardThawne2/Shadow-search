# Bharatanatyam Mudra Classification Project

This project implements a machine learning pipeline to classify mudras (hand gestures) from Bharatanatyam dance form using deep learning techniques.

## Problem Statement
Mudras (Hand gestures) in various dance forms of Bharat play an important role in the communication between the artist and the audience. They are critical to conveying the meaning and bhava to the audience. This project focuses on developing machine learning algorithms to identify mudras from full body or hand images for Bharatanatyam dance form.

## Features
- Transfer learning using MobileNetV2 for efficient training
- Data augmentation for better generalization
- Support for both hand-only and full-body image classification
- Real-time prediction capabilities
- Comprehensive evaluation metrics

## Dataset Structure
```
data/
  bharatanatyam_mudras/
    train/
      pataka/          # Open palm gesture
      tripataka/       # Three-flag gesture
      ardhapataka/     # Half-flag gesture
      kartarimukha/    # Scissors gesture
      mayura/          # Peacock gesture
      ardhachandra/    # Half-moon gesture
      arala/           # Bent gesture
      shukatunda/      # Parrot's beak gesture
      mushtika/        # Fist gesture
      sikhara/         # Peak gesture
    validation/
      [same structure as train]
    test/
      [same structure as train]
```

## Installation
1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Prepare your dataset in the structure shown above
4. Run the training script:
   ```
   python train_model.py
   ```

## Usage
- Train model: `python train_model.py`
- Evaluate model: `python evaluate_model.py`
- Real-time prediction: `python predict_realtime.py`
- Predict single image: `python predict_single.py --image_path path/to/image.jpg`

## Model Architecture
- Base: MobileNetV2 (pre-trained on ImageNet)
- Custom layers: Global Average Pooling + Dense layers
- Output: Softmax classification for 10 common Bharatanatyam mudras

## Results
The model achieves high accuracy on the validation set and can effectively classify common Bharatanatyam mudras from both hand-only and full-body images.

## References
- https://arxiv.org/html/2404.11205v1
- https://ieeexplore.ieee.org/abstract/document/10851628