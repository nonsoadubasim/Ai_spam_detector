# 💌 AI Spam Detector

## 📌 Project Overview

AI Spam Detector is a Machine Learning application that automatically
classifies SMS messages as either Spam or Ham (Legitimate Messages).

The project uses Natural Language Processing (NLP) and the Naive Bayes
algorithm to identify unwanted messages.

The application provides predictions and confidence scores through
an interactive Streamlit interface.

## 🎯 Project Objectives

- Detect unwanted spam messages.
- Identify legitimate SMS messages.
- Apply Machine Learning to real-world text classification.
- Build and deploy an interactive AI application.

## 📊 Dataset

The project uses the SMS Spam Collection Dataset.

Dataset Information:

- Original Messages: 5,572
- Cleaned Messages: 5,169
- Ham Messages: 4,516
- Spam Messages: 653
- Duplicate Messages Removed: 403

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Joblib
- Streamlit
- Jupyter Notebook

## ⚙️ Machine Learning Workflow

1. Dataset Loading
2. Data Cleaning
3. Duplicate Removal
4. Feature and Target Separation
5. Train-Test Splitting
6. TF-IDF Vectorization
7. Model Training
8. Model Evaluation
9. Model Saving
10. Streamlit Deployment

## 🤖 Model Performance

The trained model achieved the following results:

| Evaluation Metric | Result |
|-------------------|--------|
| Accuracy | 96.91% |
| Spam Precision | 100% |
| Spam Recall | 76% |
| Spam F1-Score | 86% |

### Confusion Matrix

```text
[[903   0]
 [ 32  99]]
