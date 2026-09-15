# AI Cancer Diagnosis

A Machine Learning project that predicts whether a breast tumor is **Malignant** or **Benign** using the Breast Cancer Wisconsin dataset.

The project uses a **Random Forest Classifier** with feature scaling and provides model evaluation metrics, confusion matrix analysis, ROC-AUC evaluation, and prediction probabilities.

> **Note:** This project is for educational and demonstration purposes only and is not intended for real-world medical diagnosis.

## Project Overview

This project demonstrates an end-to-end Machine Learning workflow:

* Data loading and preprocessing
* Train-test splitting using stratification
* Feature scaling using `StandardScaler`
* Random Forest model training
* Model evaluation
* Confusion matrix analysis
* ROC-AUC evaluation
* Prediction on new patient data
* Model persistence using `joblib`
* Explainability concepts explored using **SHAP** and **LIME**


## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Git & GitHub

> SHAP and LIME were explored separately as part of the project's Machine Learning explainability learning process.

## Dataset

The project uses the built-in **Breast Cancer Wisconsin dataset** provided by Scikit-learn.

| Information      | Value |
| ---------------- | ----: |
| Total Samples    |   569 |
| Features         |    30 |
| Training Samples |   455 |
| Testing Samples  |   114 |
| Malignant        |   212 |
| Benign           |   357 |

## Machine Learning Model

### Random Forest Classifier

A **Random Forest Classifier** was selected because it can:

* Handle multiple features effectively
* Capture non-linear relationships
* Provide feature importance
* Perform well on classification tasks
* Combine predictions from multiple decision trees

The trained model and feature scaler are saved using `joblib` so they can be reused for future predictions without retraining.


## Model Performance

Evaluation on the test dataset:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 95.61% |
| Precision | 95.89% |
| Recall    | 97.22% |
| F1 Score  | 96.55% |
| ROC-AUC   | 99.39% |

### Confusion Matrix

```text
[[39, 3],
 [ 2,70]]
```

The model correctly classified most samples and achieved a high recall of **97.22%**.

For a medical classification demonstration, recall is an important metric because it measures how effectively the model identifies positive cases.

## Explainable AI

Explainability techniques were also explored during the project to understand Machine Learning predictions.

### SHAP

**SHAP (SHapley Additive exPlanations)** was explored to understand how individual features influence model predictions across multiple samples.

### LIME

**LIME (Local Interpretable Model-agnostic Explanations)** was explored to understand individual predictions and identify which features contributed to a specific prediction.

These techniques provide practical exposure to **Machine Learning model interpretability and Explainable AI (XAI)**.

> The current production pipeline in this repository focuses on Random Forest training, evaluation, and prediction. SHAP and LIME were explored separately as part of the project's explainability learning.


## Project Structure

```text
AI_CANCER_DIAGNOSIS/
│
├── models/
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/psmkeshav100-max/AI_CANCER_DIAGNOSIS.git
cd AI_CANCER_DIAGNOSIS
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

### Train the Model

```bash
python src/train.py
```

This loads the Breast Cancer Wisconsin dataset, preprocesses the data, trains the Random Forest model, and saves the trained model and scaler.

### Evaluate the Model

```bash
python src/evaluate.py
```

This evaluates the trained model using classification metrics and a confusion matrix.

### Make a Prediction

```bash
python src/predict.py
```

This loads the saved model and scaler and performs a prediction using patient feature data.

## Key Learning Outcomes

Through this project, I practiced:

* Building an end-to-end Machine Learning pipeline
* Data preprocessing
* Train-test splitting
* Feature scaling
* Random Forest classification
* Model evaluation
* Confusion matrix analysis
* Precision, recall, and F1-score analysis
* ROC-AUC evaluation
* Model serialization using `joblib`
* Making predictions using saved models
* Exploring Explainable AI with SHAP and LIME
* Git and GitHub project management

## Future Improvements

* Add a web-based prediction interface
* Develop a REST API for model predictions
* Add Docker support
* Deploy the application to the cloud
* Add automated testing
* Implement CI/CD
* Add model monitoring
* Improve model explainability integration



## Author

Krishna Patel
AI/ML Engineer

If you found this project useful, consider giving the repository a star!
