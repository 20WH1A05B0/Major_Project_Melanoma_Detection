# Melanoma Detection using Deep Learning

## Overview
This repository contains the code implementation and comparison analysis for melanoma detection using Convolutional Neural Networks (CNNs) with VGG16, ResNet-50, and VGG-11 architectures. The models are trained on a dataset of skin lesion images to classify melanoma and non-melanoma lesions.

## Features
- Utilizes a pre-trained VGG16 model for feature extraction, leveraging its ability to recognize patterns in images.
- Fine-tuned specifically for melanoma detection, improving its accuracy in classifying skin lesions.
- Provides scripts for training, evaluation, and prediction, allowing users to train new models, evaluate their performance, and make predictions on new data.

## Dataset
### Data Details
* **Source:** Kaggle ([https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images))
* **Size:** 10,604 images
* **Availability:** The dataset comprises images with dimensions of 300x300 pixels.
* **Preprocessing:**
    * Resized from 300x300 pixels to 224x224 pixels to match the input size of the pre-trained VGG16 model.
    * Aspect ratio maintained during resizing to prevent image distortion.
    * Interpolation methods applied to preserve image quality during resizing.

### Dataset Balance

The dataset is balanced with a ratio of 1.07, ensuring that there is no significant class imbalance  between melanoma and non-melanoma lesions. This helps prevent the model from being biased towards the more frequent class.


## Software Requirements

### Programming Language

* Python 3.10

### Packages

* NumPy: Used for numerical computations.
* Pandas: Used for data manipulation and analysis.
* TensorFlow/PyTorch (choose one): Deep learning frameworks used for building and training the CNN models.
* Scikit-learn: Used for machine learning tasks and preprocessing.
* Matplotlib/Seaborn (choose one): Used for data visualization.

### Tool

* Jupyter Notebook: An interactive environment for developing and running Python code.

These software requirements are essential for developing, training, and evaluating machine learning models for melanoma detection using the provided dataset. Ensure that you have the specified versions of Python and the required packages installed in your development environment to run the code seamlessly.

## Usage
Clone the repository:
   ```
   git clone https://github.com/20WH1A05B0/Major_Project_Melanoma_Detection.git
   ```

## Results

| Metric | VGG16 | ResNet50 | VGG11 |
|---|---|---|---|
| Accuracy | 88.60% | 90.15% | 87.82% |
| Precision | 89.00% | 91.23% | 86.54% |
| Recall | 88.13% | 89.71% | 88.47% |
| F1-score | 88.26 | 90.47% | 87.50% |

**Table Caption:** Performance comparison of VGG16, ResNet50, and VGG11 models on the melanoma detection task.

**Figure 1: Validation and Training Accuracy vs Epochs** ([link to image](https://github.com/20WH1A05B0/Major_Project_Melanoma_Detection/blob/main/Accuracy%20vs%20Epochs.png))

This graph depicts the accuracy of each model during the training process. The x-axis represents the training epochs, and the y-axis represents the accuracy. The higher the curve for the validation accuracy, the better the model generalizes to unseen data.

**Figure 2: Validation and Training Loss vs Epochs** ([link to image](https://github.com/20WH1A05B0/Major_Project_Melanoma_Detection/blob/main/Loss%20vs%20Epochs.png))

This graph shows the training loss for each model across epochs. The loss function measures how well a model's predictions deviate from the actual labels. The goal is to minimize the loss function during training.


## References


## Authors

* [Renusree](https://github.com/20WH1A05B0)
* [Atoshi Das](https://github.com/Atoshi-Das)
* [Sindhuja](https://github.com/Sindhujaramidi)
