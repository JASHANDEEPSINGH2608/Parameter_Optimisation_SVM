# SVM Project

## Overview
This repository contains an implementation of Support Vector Machines (SVM) for a classification task. The notebook demonstrates data preprocessing, model training, hyperparameter tuning, and evaluation metrics for SVM.

## Table of Contents
1. [Features and Techniques](#features-and-techniques)
2. [Usage](#usage)
3. [Dependencies](#dependencies)
4. [Dataset](#dataset)
5. [Project Structure](#project-structure)
6. [Results](#results)


## Features and Techniques
The notebook explores the following:

- **Data Preprocessing**: Handling missing values, normalization, and feature scaling.
- **Model Training**: Building and training the SVM model on the dataset.
- **Hyperparameter Tuning**: Using techniques like grid search or cross-validation to optimize SVM parameters (e.g., `C`, `kernel`, `gamma`).
- **Evaluation Metrics**: Calculating accuracy, precision, recall, F1-score, and confusion matrix.

## Usage
To use this project:

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install the required dependencies (see below).

3. Open and run the notebook `svm.ipynb` using Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook svm.ipynb
   ```

## Dependencies
The project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Dataset
The dataset used for this project is Room_Occupancy_Data.csv. It includes:

Sensor readings (e.g., temperature, humidity, light, CO2).
Target variable: Room Occupancy Count (number of people in the room).

## Project Structure
```
SVM_Project/
|-- svm.ipynb              # Main notebook with analysis
|-- requirements.txt       # Python dependencies
|-- data/                  # Dataset used in the analysis
|-- README.md              # Project documentation
```

# Results

## Best Kernel and Parameters
- **Best Kernel:** Linear  
- **Best Accuracy:** 1.0  
- **Optimal Parameters:**  
  - Nu (C): 1.17  
  - Epsilon (Gamma): 4.03

## Performance Table

| Sample | Best Accuracy | Best Kernel | Best Nu (C) | Best Epsilon (Gamma) |
|--------|---------------|-------------|-------------|-----------------------|
| 1      | 0.99          | linear      | 8.28        | 2.83                 |
| 2      | 0.99          | linear      | 0.61        | 6.43                 |
| 3      | 1.0           | linear      | 1.17        | 4.03                 |
| 4      | 0.99          | poly        | 6.94        | 4.37                 |
| 5      | 0.99          | poly        | 5.94        | 8.56                 |
| 6      | 1.0           | linear      | 0.74        | 3.55                 |
| 7      | 0.99          | linear      | 1.20        | 6.87                 |
| 8      | 0.99          | linear      | 7.75        | 2.21                 |
| 9      | 0.99          | poly        | 3.65        | 6.32                 |
| 10     | 0.99          | linear      | 8.55        | 9.98                 |


