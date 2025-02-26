# Hand_Gesture_Recognition
# Hand Gesture Recognition using CNN

## Overview

This project builds a Convolutional Neural Network (CNN) to classify hand gestures using the **LeapGestRecog** dataset. The dataset contains 10 categories of hand gestures captured under different lighting conditions. The trained model achieves high accuracy in gesture recognition.

## Dataset

Dataset used: [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

## Features

- Loads and preprocesses the dataset
- Visualizes sample images
- Splits the dataset into training and testing sets
- Normalizes image data for better training performance
- Builds a CNN model for classification
- Trains and evaluates the model
- Saves the trained model for future use

## Installation

To run this project, install the required dependencies:

```sh
pip install tensorflow keras numpy opencv-python scikit-learn matplotlib
```

## Project Structure

```
|-- Hand Gesture Recognition
    |-- hand_gesture_recognition.ipynb  # Jupyter Notebook file
    |-- README.md                  # Project documentation
```

## Running the Project

1. Download the dataset from Kaggle and extract it.
2. Update the `data_path` variable in `hand_gesture_recognition.ipynb` to match your dataset location.
3. Open the notebook and run all cells.

## Model Architecture

The CNN model consists of:

- **Three convolutional layers** with increasing filters (64, 128, 256)
- **Max pooling layers** to reduce spatial dimensions
- **Flatten layer** to convert feature maps into a vector
- **Dense layers** for classification
- **Dropout layer** to prevent overfitting
- **Softmax activation** for multi-class classification

## Results

After training, the model achieves **near 100% accuracy** on the test set. The exact performance may vary based on hyperparameters and dataset preprocessing.

## Future Improvements

- Implement **data augmentation** to improve generalization.
- Optimize the model architecture for **faster inference**.
- Deploy the model using **Flask/Streamlit** for real-time gesture recognition.


