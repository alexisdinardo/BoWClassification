# Scene Classification with Bag of Words (BoW)

## Overview
This project implements a scene classification system using the Bag of Words (BoW) model. It extracts SIFT features from images, clusters them to form visual words, and uses machine learning models such as Support Vector Machines (SVM) and k-Nearest Neighbors (KNN) for classification.

## Features
- **Feature Extraction**: Uses SIFT descriptors to represent images.
- **BoW Representation**: Clusters SIFT features using k-means clustering.
- **Dataset Handling**: Loads and splits the scene dataset into training and testing sets.
- **Classification**: Implements SVM and KNN classifiers for scene recognition.
- **Performance Evaluation**: Measures classification accuracy on test data.


## Usage
1. **Load and Split Dataset**:
   - The script loads images from the `scenes_lazebnik/` directory.
   - Extracts SIFT features from images.
   
2. **Compute BoW Representation**:
   - Uses k-means clustering to generate visual words.
   - Represents each image as a histogram of visual words.
   
3. **Train Classifiers**:
   - Trains SVM and KNN models using the BoW representation.
   
4. **Evaluate Performance**:
   - Computes classification accuracy on test data.

