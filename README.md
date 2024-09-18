

# Credit Risk Probability of Default (PD) Model

## Project Overview

This project focuses on developing a **Credit Risk Probability of Default (PD) Model** using machine learning techniques to predict the likelihood of loan default. The model helps financial institutions assess the risk associated with loan applicants and make data-driven decisions. The model incorporates **Logistic Regression**, **Random Forest**, and **XGBoost** algorithms to provide reliable predictions and insights into credit risk. The dataset is preprocessed, balanced using **SMOTE**, and scaled to improve model accuracy. Model tracking and persistence are managed using **pickle**.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Performance Evaluation](#performance-evaluation)
7. [Feature Importance](#feature-importance)
8. [Instructions to Run](#instructions-to-run)
9. [Files Included](#files-included)
10. [Future Improvements](#future-improvements)

## Project Structure

```
.
├── credit_risk_dataset.csv         # Input dataset (loan data)
├── Credit_Risk_PD_Model.ipynb      # Jupyter notebook containing all steps of the project
├── logisticPDmodel.pkl             # Pickle file for the trained Logistic Regression model
├── RandomForesPDmodel.pkl          # Pickle file for the trained Random Forest model
├── XGBpdModel.pkl                  # Pickle file for the trained XGBoost model
├── pd_prediction.xlsx              # Output file with predictions
└── README.md                       # Project documentation
```

## Requirements

This project requires the following libraries and dependencies:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

- **Python 3.x**
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Seaborn/Matplotlib**: Data visualization.
- **scikit-learn**: Machine learning and model evaluation.
- **XGBoost**: Gradient boosting framework for high-performance training.
- **imbalanced-learn (SMOTE)**: Handling imbalanced datasets via Synthetic Minority Over-sampling Technique (SMOTE).
- **pickle**: Model serialization.

## Data Preprocessing

1. **Dataset Description**: The input dataset contains loan applicant details such as age, employment length, loan amount, and default history.

2. **Data Cleaning**: 
   
   - Removed outliers (e.g., ages above 70 years and employment lengths above 47 years).
   - Filled missing values in the `loan_int_rate` column using the median.
   - Removed unnecessary columns (e.g., `loan_grade`).

3. **Categorical Encoding**: 
   
   - Used one-hot encoding for categorical variables such as `person_home_ownership` and `loan_intent`.
   - Converted the `cb_person_default_on_file` column to a binary feature (1 for 'Y', 0 for 'N').

4. **Data Scaling**: Applied **StandardScaler** to numerical columns (`person_age`, `person_income`, `loan_amnt`, etc.) to normalize the input features.

## Feature Engineering

- Categorical variables were encoded using **one-hot encoding** for the `person_home_ownership` and `loan_intent` columns.
- Features related to the individual's credit history, loan amount, and interest rates were retained and scaled.
- The dataset was balanced using **SMOTE** to address class imbalance in the target variable (`loan_status`), creating synthetic samples for the minority class.

## Modeling

Three models were developed for the prediction of loan default risk:

1. **Logistic Regression**: A linear model used for binary classification.
2. **Random Forest**: An ensemble method for classification using multiple decision trees.
3. **XGBoost**: A gradient boosting algorithm known for high accuracy and performance.

Each model was trained on the preprocessed and balanced dataset, and their performance was evaluated using precision, recall, and F1 scores.

### Model Persistence

All trained models were serialized using **pickle** to allow for future use in real-time predictions:

- `logisticPDmodel.pkl`
- `RandomForesPDmodel.pkl`
- `XGBpdModel.pkl`

## Performance Evaluation

The performance of the models was evaluated on a test set (20% of the dataset) using the following metrics:

- **Precision**: Measure of accuracy when a positive prediction is made.
- **Recall**: Measure of how well the model captures actual defaults.
- **F1-Score**: Harmonic mean of precision and recall to balance both aspects.
- **Accuracy**: Overall correctness of the predictions.

The following are sample results from the classification report for the **Logistic Regression** model:

```
              precision    recall  f1-score   support

           0       0.89      0.91      0.90      4962
           1       0.90      0.88      0.89      4879

    accuracy                           0.89      9841
   macro avg       0.89      0.89      0.89      9841
weighted avg       0.89      0.89      0.89      9841
```

Similar evaluation metrics were used for **RandomForest** and **XGBoost**, both showing high predictive accuracy.

## Feature Importance

For interpretability and feature insights:

- **Logistic Regression**: The model coefficients were extracted to evaluate the importance of features.
- **RandomForest/XGBoost**: Feature importance scores were derived from the trained models to determine which variables had the most influence on the prediction.

The feature importance results help in understanding the significant factors contributing to loan default risk, providing insights for financial decision-making.

## Instructions to Run

1. **Clone or Download the Project**:
   
   ```bash
   git clone https://github.com/your-repo/credit-risk-pd-model.git
   cd credit-risk-pd-model
   ```

2. **Install Dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   Open the **Credit_Risk_PD_Model.ipynb** file in **Jupyter Notebook** or **Google Colab** to see the entire process of data preprocessing, model training, and evaluation.

4. **Run Pre-trained Models**:
   The trained models can be loaded using the following code:
   
   ```python
   import pickle
   with open('logisticPDmodel.pkl', 'rb') as f:
       model = pickle.load(f)
   predictions = model.predict(test_data)
   ```

5. **Evaluate Models**:
   Use the saved models to make predictions and evaluate their performance on new datasets.

## Files Included

- `credit_risk_dataset.csv`: The dataset containing loan applicant details.
- `Credit_Risk_PD_Model.ipynb`: The main Jupyter notebook with all code and steps for the project.
- `logisticPDmodel.pkl`, `RandomForesPDmodel.pkl`, `XGBpdModel.pkl`: Pickled models for Logistic Regression, Random Forest, and XGBoost, respectively.
- `pd_prediction.xlsx`: Output file containing predictions on the test dataset.

## Future Improvements

- **Hyperparameter Tuning**: Perform grid search or random search to further optimize model performance.
- **Feature Selection**: Apply advanced feature selection techniques to remove irrelevant features.
- **Model Deployment**: Implement the model in a web application or API for real-time use by financial institutions.
- **Advanced Imbalance Handling**: Explore techniques beyond SMOTE, such as **ADASYN** or **Cost-Sensitive Learning**, to handle class imbalance more effectively.
