# fatigue_prediction_steel
# Prediction of Mechanical Properties of Steels Using Machine Learning

## Project Overview
This project leverages machine learning techniques to predict the mechanical properties of steels, such as fatigue strength, tensile strength, hardness, and ductility. By analyzing the chemical composition and heat treatment conditions, this project aims to provide an efficient and cost-effective alternative to traditional testing methods.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Usage](#usage)
- [Conclusions](#conclusions)
- [References](#references)

## Introduction
Steel is a versatile material whose mechanical properties determine its suitability for various engineering applications. Traditional methods for determining these properties are time-consuming and expensive. This project uses machine learning models, such as Linear Regression, K-Nearest Neighbors (KNN), and CatBoost, to predict steel properties efficiently.

## Features
- Predict mechanical properties of steels based on chemical composition and heat treatment conditions.
- Utilizes advanced ML models to improve prediction accuracy.
- Provides insights into the relationships between steel properties and composition/treatment.

## Dataset
The dataset used for this project is the Fatigue Dataset for Steel, obtained from the National Institute of Material Science (NIMS) MatNavi database. It contains:
- Chemical compositions (e.g., Carbon, Silicon, Manganese).
- Heat treatment parameters (e.g., Normalizing Temperature, Tempering Time).
- Mechanical properties such as fatigue strength.

### Data Attributes
1. **Chemical Composition**: Carbon (C), Silicon (Si), Manganese (Mn), etc.
2. **Heat Treatment Parameters**: Normalizing Temperature (NT), Tempering Time (Tt), etc.
3. **Mechanical Property**: Fatigue strength.

## Libraries Used
The project relies on the following Python libraries:
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning model implementation and evaluation.
- **CatBoost**: For advanced gradient boosting models.

## Machine Learning Models
### 1. Linear Regression
- Assumes a linear relationship between input features and target variables.
- Achieved an R^2 score of 0.934 for the test dataset.

### 2. K-Nearest Neighbors (KNN)
- A non-parametric method that predicts based on the 'k' nearest neighbors.
- Achieved an R^2 score of 0.711 for the test dataset.

### 3. CatBoost
- Gradient boosting algorithm optimized for categorical data.
- Achieved an R^2 score of 0.924 for the test dataset.

## Results
| Model                | R^2 Score (Test) | RMSE (Test) |
|----------------------|------------------|-------------|
| Linear Regression    | 0.934            | 39.07       |
| K-Nearest Neighbors  | 0.711            | 82.64       |
| CatBoost             | 0.924            | 42.72       |

## Usage
1. **Data Preparation**:
   - Ensure the dataset is properly preprocessed and cleaned.
   - Split the data into training and testing sets.

2. **Model Training**:
   - Train the models using the training data.
   - Optimize hyperparameters using grid search or cross-validation.

3. **Prediction**:
   - Use trained models to predict the mechanical properties of steel for new compositions and treatments.

4. **Evaluation**:
   - Evaluate model performance using metrics like R^2 score and RMSE.

## Conclusions
- **Linear Regression** performed the best, achieving the highest R^2 score.
- **CatBoost** showed competitive performance and is suitable for handling complex relationships.
- **KNN** underperformed compared to other models but still provides useful insights for certain applications.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [CatBoost Documentation](https://catboost.ai/)
- National Institute of Material Science (NIMS) MatNavi Database
- Online resources on machine learning and materials science
