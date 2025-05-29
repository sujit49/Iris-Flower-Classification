# Iris Flower Classification using K-Nearest Neighbors (KNN)

This project demonstrates a machine learning pipeline for classifying species of Iris flowers (Setosa, Versicolor, Virginica) based on their sepal and petal measurements. It uses the classic Iris dataset, performs exploratory data analysis (EDA), trains a K-Nearest Neighbors (KNN) classifier, evaluates its performance, and saves the trained model.

## Features

-   **Data Loading & Cleaning:** Loads the Iris dataset from a CSV file (`Iris.csv`) and performs basic cleaning (e.g., dropping unnecessary 'Id' column, standardizing column names).
-   **Exploratory Data Analysis (EDA):**
    -   Displays descriptive statistics of the dataset.
    -   Shows the distribution of species.
    -   Visualizes relationships between features using a pairplot.
    -   Displays a heatmap of feature correlations.
-   **Data Preprocessing:**
    -   Separates features (X) and the target variable (y - species).
    -   Splits the data into training and testing sets.
    -   Standardizes feature values using `StandardScaler` to ensure all features contribute equally to model training.
-   **Model Training:**
    -   Trains a K-Nearest Neighbors (KNN) classifier with `n_neighbors=5`.
-   **Model Evaluation:**
    -   Makes predictions on the test set.
    -   Generates and displays a confusion matrix to assess classification accuracy for each species.
    -   Prints a detailed classification report showing precision, recall, F1-score, and support.
    -   Visualizes the confusion matrix as a heatmap.
-   **Model Persistence:**
    -   Saves the trained KNN model to a file (`iris_knn_model.pkl`) using `joblib` for future use without retraining.
    -   Demonstrates how to load the saved model and make predictions.

## How to Use

### Prerequisites

-   Python 3.x
-   The following Python libraries:
    -   `numpy`
    -   `pandas`
    -   `seaborn`
    -   `matplotlib`
    -   `scikit-learn`
    -   `joblib`

### Installation

1.  **Clone the repository or download the files (`main.py` and `Iris.csv`).**
2.  **Install the required libraries:**
    Open your terminal or command prompt and run:
    ```bash
    pip install numpy pandas seaborn matplotlib scikit-learn joblib
    ```

### Running the Script

1.  Ensure the `Iris.csv` file is in the same directory as `main.py`.
2.  Navigate to the directory containing `main.py` in your terminal.
3.  Run the script using the command:
    ```bash
    python main.py
    ```
4.  The script will output:
    -   The first five rows of the dataset.
    -   Renamed column names.
    -   Dataset statistics and species distribution.
    -   A pairplot and correlation heatmap (displayed in separate windows).
    -   The confusion matrix (text and heatmap).
    -   The classification report.
    -   Predictions using the loaded model for the first 5 test samples.
    -   A file named `iris_knn_model.pkl` will be created in the same directory, containing the trained model.

## File Structure

-   `main.py`: The main Python script containing all the code for data loading, EDA, preprocessing, model training, evaluation, and saving.
-   `Iris.csv`: The dataset file containing the Iris flower measurements and species.
-   `iris_knn_model.pkl` (Output): The saved trained K-Nearest Neighbors model.

## Workflow Overview

1.  **Load Data:** The `Iris.csv` dataset is loaded into a pandas DataFrame.
2.  **Inspect & Clean:** Initial inspection (e.g., `df.head()`, `df.describe()`) and cleaning (dropping 'Id', renaming columns) are performed.
3.  **Visualize:** `seaborn.pairplot` and `seaborn.heatmap` are used to understand feature distributions and correlations.
4.  **Prepare Data for Modeling:**
    -   Features (X) and target (y) are separated.
    -   Data is split into 80% training and 20% testing sets.
    -   `StandardScaler` is applied to scale numerical features.
5.  **Train KNN Model:** A `KNeighborsClassifier` is initialized and trained on the scaled training data.
6.  **Evaluate Model:** Predictions are made on the scaled test data. Performance is measured using `confusion_matrix` and `classification_report`.
7.  **Save Model:** The trained `knn` model is serialized and saved to `iris_knn_model.pkl` using `joblib.dump()`.
8.  **Load Model (Example):** The script also demonstrates how to load the saved model using `joblib.load()` and use it for predictions.

## Potential Enhancements

-   **Hyperparameter Tuning:** Use techniques like GridSearchCV or RandomizedSearchCV to find the optimal `n_neighbors` for the KNN model.
-   **Comparison with Other Models:** Implement and compare the performance of other classification algorithms (e.g., Logistic Regression, Support Vector Machines, Decision Trees, Random Forest).
-   **Cross-Validation:** Employ cross-validation during training for a more robust evaluation of model performance.
-   **Interactive Predictions:** Build a simple interface (e.g., using Streamlit or Flask) to allow users to input flower measurements and get species predictions from the loaded model.
-   **More Detailed EDA:** Expand the exploratory data analysis section with more plots or statistical tests.
