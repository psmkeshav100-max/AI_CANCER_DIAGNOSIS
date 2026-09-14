#  AI Cancer Diagnosis

A Machine Learning project that predicts whether a breast tumor is **Malignant** or **Benign** using the Breast Cancer Wisconsin dataset.

The project uses a **Random Forest Classifier** with feature scaling and provides model evaluation metrics along with prediction probabilities.


##  Project Overview

This project demonstrates an end-to-end Machine Learning workflow:

* Data loading and preprocessing
* Train-test splitting
* Feature scaling using `StandardScaler`
* Random Forest model training
* Model evaluation
* Prediction on new patient data
* Model persistence using `joblib`
* Explainable AI using **SHAP** and **LIME**


##  Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* SHAP
* LIME
* Joblib
* Git & GitHub


##  Dataset

The project uses the built-in **Breast Cancer Wisconsin dataset** from Scikit-learn.

| Information      | Value |
| ---------------- | ----: |
| Total Samples    |   569 |
| Features         |    30 |
| Training Samples |   455 |
| Testing Samples  |   114 |
| Malignant        |   212 |
| Benign           |   357 |


##  Machine Learning Model

### Random Forest Classifier

Random Forest was selected because it can:

* Handle multiple features effectively
* Capture non-linear relationships
* Provide feature importance
* Perform well on classification tasks


##  Model Performance

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

The model correctly classified most malignant and benign cases while maintaining a high recall.

> **Note:** This project is for educational and demonstration purposes only and is not intended for real-world medical diagnosis.


##  Explainable AI

The project also explores model explainability using:

### SHAP

Used to understand how features influence model predictions across the dataset.

### LIME

Used to explain individual predictions and identify which features contributed to a specific prediction.

These techniques help make Machine Learning predictions more interpretable.



##  Project Structure

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

##  Installation

Clone the repository:

```bash
git clone https://github.com/psmkeshav100-max/AI_CANCER_DIAGNOSIS.git
cd AI_CANCER_DIAGNOSIS
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


##  Run the Project

### Train the model

```bash
python src/train.py
```

### Evaluate the model

```bash
python src/evaluate.py
```

### Make a prediction

```bash
python src/predict.py
```

##  Key Learning Outcomes

Through this project, I practiced:

* Building a complete ML pipeline
* Data preprocessing
* Feature scaling
* Classification
* Model evaluation
* Confusion matrix analysis
* ROC-AUC evaluation
* Model serialization
* Making predictions using saved models
* Explainable AI with SHAP and LIME
* Git and GitHub project management


##  Future Improvements

* Add a web-based prediction interface
* Deploy the model as an API
* Add Docker support
* Deploy the application to the cloud
* Add automated testing and CI/CD
* Improve model monitoring


##  Author

Krishna Patel
AI/ML Engineer


If you found this project useful, consider giving the repository a star!
