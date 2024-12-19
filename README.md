# Anomaly-Detection-in-Financial-Transactions
For LSTM:
  The LSTM model was trained for 80 epochs on the credit card data present in Kaggle at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. The data had PCA        already applied to it. Since the data was highly unbalanced, we used SMOTE to make it balanced. The trained weights can be found in the file                           "lstm_model_weights.pth". Our LSTM model considers 10 time steps at a single time. We were able to achieve a test accuracy of 99.88%. Since PCA was applied to this    dataset we weren't able to apply any explainable AI techniques to it since the columns are not meaningful.


# Fraud Detection with Autoencoder and LSTM (File - ML_Final_project.ipynb

This repository contains a machine learning project aimed at detecting fraudulent financial transactions. The project leverages an autoencoder for feature extraction and dimensionality reduction, followed by an LSTM-based classifier to distinguish between legitimate and fraudulent transactions. The dataset used is the **PaySim** dataset from Kaggle.

## Project Overview

Financial fraud detection is a critical task for banks, payment gateways, and financial institutions. This project demonstrates an end-to-end pipeline, including data preprocessing, feature engineering using an autoencoder, and sequence modeling via LSTM layers to predict fraudulent activities.

### Key Steps

1. **Data Loading and Exploration:**  
   - Downloading and loading the PaySim dataset from Kaggle.
   - Exploratory Data Analysis (EDA) to understand data distribution, check for missing values, and assess class imbalance.
   
2. **Data Preprocessing:**  
   - Encoding categorical features using one-hot encoding.
   - Scaling numerical features for better model convergence.
   - Splitting the data into training and testing sets.
   
3. **Feature Extraction with Autoencoder:**  
   - Using a deep autoencoder to learn latent representations of non-fraudulent transactions.
   - Reducing dimensionality and noise, resulting in compressed feature embeddings.
   
4. **Modeling with LSTM:**  
   - Feeding encoded features into an LSTM-based neural network.
   - Training the LSTM model to classify transactions as fraudulent or legitimate.
   
5. **Model Evaluation:**  
   - Evaluating model performance using metrics such as precision, recall, F1-score, PR-AUC, and ROC-AUC.
   - Analyzing confusion matrices and plotting Precision-Recall curves.
   
6. **Explainability with SHAP:**  
   - Using SHAP (SHapley Additive exPlanations) to interpret model predictions.
   - Understanding feature contributions to model outputs on selected samples.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- SHAP

You can install the required packages using:

```bash
pip install -r requirements.txt
