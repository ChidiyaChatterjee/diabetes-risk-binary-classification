# Diabetes Risk Binary Classification

A beginner-friendly machine-learning project that uses health measurements to classify diabetes outcome as a binary label. The workflow is in a Jupyter notebook and covers data exploration, visualization, preprocessing, model comparison, tuning, and evaluation. There is currently no web app or user interface.

> **Educational use only:** This project is not a medical diagnostic tool. Do not use its predictions for health or treatment decisions.

## Project Goal

Predict the `Outcome` class from eight health-related input features:

- `0`: No diabetes
- `1`: Diabetes

The model inputs are `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.

## Dataset

The notebook expects a local file named `diabetes.csv`. The dataset is not included in this repository, so obtain it separately and put it in the project directory before running the notebook.

The expected dataset has **768 rows**, eight numeric input features, and one binary target column. It must contain these columns:

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and `Outcome`.

The file `diabetes.csv` is ignored by Git. Generated model files are also excluded, so neither the data nor the trained model binaries are uploaded with the source.

## What Is in the Notebook

The notebook walks through a complete introductory classification workflow:

1. Load the CSV and inspect its shape, columns, types, and summary statistics.
2. Check zero values and replace zero-coded missing measurements in selected columns with the median.
3. Explore class balance and visualize feature distributions, correlations, and potential outliers.
4. Remove outliers from selected features using the interquartile range (IQR) rule.
5. Split the data into training and test sets, then standardize the input features.
6. Compare Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, and XGBoost using five-fold cross-validation.
7. Tune Logistic Regression and Gradient Boosting with recall as the model-selection score.
8. Evaluate models with accuracy, precision, recall, F1-score, and confusion matrices.
9. Save the tuned Logistic Regression model and feature scaler for local use.

## Model Results

In the notebook's recorded cross-validation run, **Gradient Boosting** had the highest mean accuracy: **0.7746**. The notebook also tunes Logistic Regression and Gradient Boosting using recall, then saves the tuned Logistic Regression estimator and the scaler. These are different selection steps: the highest accuracy in the initial comparison is not the same as the model currently saved by the notebook.

Results can vary when the data, preprocessing, package versions, or random split changes. See the notebook's evaluation output for the detailed classification reports and confusion matrices.

## Run locally

Use Python 3.10 or newer. Clone or download the repository, place your local `diabetes.csv` in its directory, then create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and launch JupyterLab:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `Diabetes-Binary-classification.ipynb` and run all cells in order. The notebook loads the local CSV, displays its analysis and evaluation results, then creates `diabetes_model.pkl` and `diabetes_scaler.pkl` in the project directory. Those generated files remain local and are ignored by Git.

## Project Files

- `Diabetes-Binary-classification.ipynb` - notebook with the analysis and modeling workflow.
- `requirements.txt` - Python dependencies.
- `.gitignore` - excludes the local dataset, generated model files, and local environment/editor files.
- `diabetes.csv` - required local dataset; intentionally not tracked by Git.

## Notes and Limitations

- This project is for learning and experimentation, not clinical use.
- The classes are imbalanced, so accuracy alone does not describe model performance; inspect diabetes-class recall, precision, F1-score, and the confusion matrix.
- The notebook currently imputes missing measurements and removes outliers before the train/test split. This can leak information into evaluation. For a more reliable estimate, fit preprocessing only on training folds using a scikit-learn pipeline and stratify the train/test split.
- The project is notebook-only. A user interface has not been implemented.