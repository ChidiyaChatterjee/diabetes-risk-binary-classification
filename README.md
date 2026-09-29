# Diabetes Risk Binary Classification

A machine-learning project exploring binary diabetes classification. The current project workflow is implemented in a Jupyter notebook; UI work is separate. The dataset is kept local and is not included in this repository.

## Run the notebook

Use Python 3.10 or newer. From the project directory, install the dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Place your local `diabetes.csv` file in the project directory, then open `Diabetes-Binary-classification.ipynb` and run its cells from top to bottom. `diabetes.csv` is excluded from Git.

## Notebook workflow

- Explore the dataset and inspect zero-coded missing measurements.
- Replace zero values in selected measurement columns with their column medians.
- Visualize class balance, feature distributions, correlations, and outliers.
- Compare classification models, tune selected models, and evaluate them on a held-out test set.
- Save the tuned Logistic Regression model and scaler as local pickle files.

The saved model and scaler are generated locally and are excluded from Git. Run the notebook to recreate them.

## Important limitations

This is an educational classification project, not a medical diagnostic tool. Its results should not be used for clinical decisions. Preprocessing and outlier filtering currently happen before the train/test split; for a more rigorous evaluation, fit preprocessing steps on training data only, preferably in a scikit-learn pipeline, and use stratified splitting.