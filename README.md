# DenseNet Retinal Disease Classification

A robust deep learning implementation for retinal disease classification using DenseNet121 architecture. This project implements a state-of-the-art approach to classify retinal diseases using transfer learning with the DenseNet121 model pre-trained on ImageNet.

## Project Structure

```
scripts/
├── main.py              # Entry point for training and evaluation
├── model.py             # DenseNet model architecture and trainer implementation
├── training.py          # Training pipeline and utilities
├── dataset.py          # Dataset handling and preprocessing
├── balance_dataset.py   # Dataset balancing utilities
├── explore_dataset.py   # Dataset exploration and visualization
├── analyze_results.py   # Results analysis and visualization
├── comprehensive_analysis.py  # Detailed model performance analysis
└── run_training.sh      # Training execution script
```

## Features

- Transfer learning using DenseNet121 architecture
- Robust training pipeline with early stopping
- Learning rate scheduling
- Comprehensive metrics tracking and visualization
- Dataset balancing and preprocessing
- Detailed performance analysis tools

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- wandb (for experiment tracking)
- tqdm

## Usage

1. Setup your environment:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn wandb tqdm
```

2. Prepare your dataset:
```bash
python balance_dataset.py
python explore_dataset.py  # Optional: to visualize dataset statistics
```

3. Train the model:
```bash
./run_training.sh
# or
python main.py --config config.yaml
```

4. Analyze results:
```bash
python analyze_results.py
python comprehensive_analysis.py
```

## Model Architecture

The implementation uses DenseNet121 as the backbone, with custom modifications:
- Pre-trained weights from ImageNet
- Custom classifier head with dropout for regularization
- Cross-entropy loss function
- Adam optimizer with learning rate scheduling

## Training Pipeline

The training process includes:
- Data augmentation for improved generalization
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Model checkpointing
- Comprehensive metrics tracking

## Results Analysis

The project includes tools for:
- Confusion matrix visualization
- ROC curve analysis
- Classification reports
- Performance metrics tracking
- Error analysis

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - feel free to use this code for your own projects!
