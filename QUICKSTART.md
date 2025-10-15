# Quick Start Guide

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create dataset structure:
```bash
python prepare_dataset.py
```

## Dataset Preparation

1. Collect images for each mudra class:
   - pataka (Open palm gesture)
   - tripataka (Three-flag gesture)
   - ardhapataka (Half-flag gesture)
   - kartarimukha (Scissors gesture)
   - mayura (Peacock gesture)
   - ardhachandra (Half-moon gesture)
   - arala (Bent gesture)
   - shukatunda (Parrot's beak gesture)
   - mushtika (Fist gesture)
   - sikhara (Peak gesture)

2. Place images in the respective directories:
   ```
   data/bharatanatyam_mudras/
     train/[mudra_name]/
     validation/[mudra_name]/
     test/[mudra_name]/
   ```

## Training

Train the model:
```bash
python train_model.py
```

## Evaluation

Evaluate the trained model:
```bash
python evaluate_model.py
```

## Prediction

### Single Image Prediction
```bash
python predict_single.py --image_path path/to/image.jpg
```

### Real-time Prediction
```bash
python predict_realtime.py
```

## Project Structure

```
25157/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.py                # Configuration settings
├── utils.py                 # Utility functions
├── prepare_dataset.py       # Dataset preparation script
├── train_model.py          # Model training script
├── evaluate_model.py       # Model evaluation script
├── predict_single.py       # Single image prediction
├── predict_realtime.py     # Real-time webcam prediction
├── data/                   # Dataset directory
│   └── bharatanatyam_mudras/
│       ├── train/          # Training images
│       ├── validation/     # Validation images
│       └── test/          # Test images
├── models/                 # Trained models
└── results/               # Training results and plots
```