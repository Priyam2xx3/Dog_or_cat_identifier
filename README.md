# 🐶🐱 Detection of Dog or Cat using CNN

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images as either a **dog** or a **cat**. Using deep learning techniques, the model learns visual features from a dataset of labeled animal images. With an achieved accuracy of approximately **85%**, the model demonstrates strong performance in binary image classification tasks.

## 🚀 Objective

To build a deep learning model capable of accurately identifying whether an input image contains a **cat** or a **dog**, using a supervised learning approach.

## 🧠 Key Features

- Deep Learning-based binary image classifier
- Implemented using **CNN architecture**
- Achieved **~85% accuracy** on validation data
- Built with **TensorFlow/Keras**
- Preprocessing and augmentation for robust performance

## 📂 Dataset

- Dataset: [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)  
- Images: 25,000 labeled images (12,500 dogs and 12,500 cats)
- Format: JPEG images

## 🔧 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV / PIL
- NumPy, Pandas, Matplotlib

## 🏗️ Model Architecture

- Input Layer: 150x150 RGB image
- 3 Convolutional Layers with MaxPooling
- Flattening Layer
- Fully Connected Dense Layers with Dropout
- Output Layer with Sigmoid activation

## 📈 Performance

- **Training Accuracy:** ~87%
- **Validation Accuracy:** ~85%
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam

## 🖼️ Sample Predictions

| Image | Predicted Label |
|-------|------------------|
| ![cat](sample_images/cat.jpg) | Cat |
| ![dog](sample_images/g.jpg) | Dog |

## 💻 How to Run

```bash
# Clone this repository
git clone https://github.com/your-username/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Predict on a new image
python predict.py --image path_to_image.jpg
