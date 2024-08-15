# Convolutional_Neural_Network
# CIFAR-10 CNN Classification

This repository contains a convolutional neural network (CNN) built using TensorFlow and Keras for image classification on the CIFAR-10 dataset. The model is designed to classify images into one of ten classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

## Model Architecture

The CNN model used in this project consists of the following layers:

- **3 Convolutional Layers**: Each followed by Batch Normalization, MaxPooling, and Dropout.
- **Fully Connected Layer**: A Dense layer with 256 neurons and ReLU activation.
- **Output Layer**: A Dense layer with 10 neurons (one for each class) and softmax activation.

## Training

- **Data Augmentation**: The training images are augmented using rotation, width/height shifts, and horizontal flips.
- **Optimizer**: Adam optimizer is used for training the model.
- **Loss Function**: Categorical Crossentropy is used as the loss function.
- **Callbacks**: The model is trained with `ModelCheckpoint` to save the best model and `EarlyStopping` to prevent overfitting.

## Evaluation

The model is evaluated on the test dataset using accuracy and a confusion matrix to visualize performance across classes. A classification report is also generated to show precision, recall, and F1-score for each class.

## Requirements

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
