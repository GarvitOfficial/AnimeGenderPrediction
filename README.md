# Anime Gender Prediction

A deep learning project that implements a Convolutional Neural Network (CNN) for binary classification of anime character images based on gender (Masculine vs. Feminine).

## 📋 Overview

This project uses TensorFlow and Keras to build an image classification model that can predict the gender of anime characters from their images. The model is trained on a dataset of anime character images labeled as either masculine (0) or feminine (1).

## 🎯 Problem Statement

Binary image classification task to determine whether an anime character is masculine or feminine based on visual features.

## 🏗️ Model Architecture

The model uses a Convolutional Neural Network (CNN) with the following architecture:

- **Input Layer**: 128x128 RGB images
- **Convolutional Layers**: Multiple Conv2D layers for feature extraction
- **Pooling Layers**: MaxPooling2D layers for dimensionality reduction
- **Dense Layers**: Fully connected layers for classification
- **Dropout Layers**: For regularization to prevent overfitting
- **Output Layer**: Single neuron with sigmoid activation for binary classification

## 📊 Dataset

The dataset consists of:
- **Training Set**: 4,800 images
- **Validation Set**: 600 images
- **Test Set**: 600 images

Images are organized in separate directories (`train_images`, `val_images`, `test_images`) with corresponding CSV files containing image IDs and labels.

## 🔧 Technologies Used

- **Python 3.x**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib** (for visualization)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/GarvitOfficial/AnimeGenderPrediction.git
cd AnimeGenderPrediction
```

2. Install required dependencies:
```bash
pip install tensorflow numpy pandas matplotlib
pip install protobuf==3.20.3
```

## 🚀 Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook anime-gender-prediction.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the data
   - Build the CNN model
   - Train the model (15 epochs)
   - Visualize training and validation metrics
   - Generate predictions on test data
   - Create submission file

### Key Configuration Parameters

- **Image Size**: 128x128 pixels
- **Batch Size**: 32
- **Epochs**: 15
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

## 📈 Data Augmentation

The training data is augmented using the following techniques:
- Rotation (up to 20 degrees)
- Width and height shifts (20%)
- Shear transformation (20%)
- Zoom (20%)
- Horizontal flip
- Rescaling (normalization to 0-1 range)

## 📊 Model Training

The model is trained with:
- **Training data**: 4,800 images with augmentation
- **Validation data**: 600 images (no augmentation)
- **Metrics tracked**: Accuracy and Loss (both training and validation)

## 📁 Project Structure

```
AnimeGenderPrediction/
│
├── anime-gender-prediction.ipynb    # Main Jupyter notebook
└── README.md                         # Project documentation
```

## 📝 Output

The notebook generates:
- Training/validation accuracy and loss plots
- `submission.csv` file containing predictions for the test set

## 🎓 Model Performance

The model's performance can be evaluated through:
- Training and validation accuracy curves
- Training and validation loss curves
- Final validation accuracy achieved during training

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👤 Author

**Garvit**
- GitHub: [@GarvitOfficial](https://github.com/GarvitOfficial)

## 📄 License

This project is part of a Kaggle competition for anime gender classification.

## 🙏 Acknowledgments

- Dataset source: Kaggle Anime Gender Classification Competition
- Built with TensorFlow and Keras frameworks
