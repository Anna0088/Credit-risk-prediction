# Credit Risk Analysis and Prediction

This project focuses on analyzing and predicting credit risk using a machine learning approach. The goal is to build a model that can accurately assess the likelihood of loan defaults based on various factors such as age, income, employment history, and loan characteristics.

## Project Steps

1. **Data Loading and Cleaning:** Load the credit risk dataset and perform data cleaning tasks such as:
    - Handling missing values
    - Removing outliers
    - Correcting data inconsistencies

2. **Exploratory Data Analysis (EDA):** Explore the dataset to gain insights into the data distribution, relationships between variables, and identify potential patterns related to credit risk.

3. **Feature Engineering:** Transform and select relevant features to improve model performance. This may include:
    - Creating new features
    - Encoding categorical variables
    - Scaling numerical features

4. **Data Balancing:** Address class imbalance issues using techniques like SMOTE to ensure fair representation of different loan status categories.

5. **Model Training:** Train various machine learning models, including:
    - Logistic Regression
    - Random Forest
    - XGBoost

6. **Model Evaluation:** Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

7. **Model Selection:** Select the best performing model based on evaluation results.

8. **Model Deployment:** Save the trained model for future use and potentially deploy it as part of a credit risk assessment system.

## Files

- `credit_risk_dataset.csv`: The original credit risk dataset.
- `pd_prediction.xlsx`: The dataset with predictions from different models.
- `logisticPDmodel.pkl`: Saved Logistic Regression model.
- `RandomForesPDmodel.pkl`: Saved Random Forest model.
- `XGBpdModel.pkl`: Saved XGBoost model.

## Libraries Used

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- imblearn
- xgboost

## Future Work

- Explore more advanced feature engineering techniques.
- Fine-tune model hyperparameters for optimal performance.
- Develop a user interface for interacting with the model.
- Deploy the model in a real-world setting.
