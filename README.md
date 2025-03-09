# Understanding-Pipelines-in-sklearn

# Understanding Pipeline in Scikit-learn

This repository provides examples and explanations for using `Pipeline` in Scikit-learn. Pipelines simplify machine learning workflows by combining data transformation steps with an estimator into a single unit.

## Table of Contents

1.  [Introduction to Pipeline](#1-introduction-to-pipeline)
2.  [Examples](#2-examples)
    * [2.1: Example 1: Linear Regression on Sinusoids](#21-example1-linear-regression-on-sinusoids)
    * [2.2: Example 2: Cancer Dataset](#22-example2-cancer-dataset)
    * [2.3: Example 3: Titanic Dataset](#23-example3-titanic-dataset)
        * [2.3.1: Library and Data Import](#231-library-and-data-import)
        * [2.3.2: Define Pipelines](#232-define-pipelines)
        * [2.3.3: Define Final Pipeline and Predict](#233-define-final-pipeline-and-predict)
    * [2.4: Example 4: Iris Dataset](#24-example4-iris-dataset)
3.  [References](#3-references)

## 1. Introduction to Pipeline

Machine learning workflows often involve multiple steps, including data preprocessing (feature transformation, dimensionality reduction, scaling) and model training. Scikit-learn's `Pipeline` simplifies this process by chaining these steps together. This allows for cleaner code, easier hyperparameter tuning, and prevention of data leakage.

## 2. Examples

### 2.1: Example 1: Linear Regression on Sinusoids

This example demonstrates how to use a pipeline to perform linear regression on a sinusoidal dataset. It showcases feature transformation using `FunctionTransformer` and combining it with a `LinearRegression` model.

* **Generate Training Data:** Creates a sinusoidal dataset with added noise.
* **Generate Test Data:** Creates a separate test dataset.
* **Linear Regression (without feature engineering):** Shows the poor performance of linear regression without feature transformation.
* **Using Pipeline for Linear Regression on Transformed Feature:** Demonstrates how to use a pipeline to apply a sine transformation and then perform linear regression.

### 2.2: Example 2: Cancer Dataset

This example uses the breast cancer dataset to illustrate how to combine `MinMaxScaler` with an `SVC` (Support Vector Classifier) in a pipeline. It also shows how to use `GridSearchCV` to tune the hyperparameters of the pipeline.

* **Data Loading and Exploration:** Loads the dataset and performs basic analysis.
* **Data Splitting:** Splits the dataset into training and testing sets.
* **Building Pipeline:** Creates a pipeline with scaling and SVM.
* **Pipeline Evaluation:** Evaluates the pipeline's performance.
* **Hyperparameter Tuning:** Uses `GridSearchCV` to find the best hyperparameters.

### 2.3: Example 3: Titanic Dataset

This example uses the Titanic dataset to demonstrate a more complex pipeline with `ColumnTransformer` for handling different data types.

* **2.3.1: Library and Data Import:** Imports necessary libraries and loads the Titanic dataset.
* **2.3.2: Define Pipelines:** Creates separate pipelines for numerical and categorical features, then combines them using `ColumnTransformer`.
* **2.3.3: Define Final Pipeline and Predict:** Creates a final pipeline with the preprocessor and a `RandomForestClassifier`, fits the model, and generates predictions for submission.

### 2.4: Example 4: Iris Dataset

This example uses the iris dataset to show a pipeline using `StandardScaler`, `PCA` and `LogisticRegression`.

* **Load the iris dataset:** Loads the dataset and splits it into training and testing sets.
* **Create a pipeline:** Creates a pipeline with scaling, PCA, and logistic regression.
* **Fit the pipeline:** Fits the pipeline to the training data.
* **Evaluate the pipeline:** Evaluates the pipeline on the test data.

## 3. References

* [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
* [Scikit-learn ColumnTransformer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
* [Scikit-learn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
* [Python Guides - Scikit-learn Pipeline](https://pythonguides.com/scikit-learn-pipeline/)
* [Scikit-learn Example: Pipeline with Digits Dataset](https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py)
