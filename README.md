📊 Loan Approval Prediction using Machine Learning
This project aims to build a machine learning model that predicts whether a loan application will be approved based on various applicant and loan attributes. The dataset is preprocessed, analyzed, and used to train multiple classifiers including Logistic Regression, Random Forest, K-Nearest Neighbors, and Support Vector Machines.

📁 Project Overview
The notebook includes the following key stages:

Mounting Google Drive to access the dataset

Exploratory Data Analysis (EDA) and Visualization

Data Cleaning and Preprocessing

Feature Encoding and Missing Value Imputation

Model Training using multiple classifiers

Model Evaluation using Accuracy Score

📌 Dataset
Source: Google Drive

File: LoanApprovalPrediction.csv

Attributes: Information about applicants like gender, marital status, education, income, loan amount, and loan status.

🔧 Dependencies
Make sure to install the required libraries before running the notebook:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
📊 Data Preprocessing
Dropped the Loan_ID column as it's unique for each record.

Categorical features were encoded using Label Encoding.

Visualized class distributions and correlations using seaborn.

Missing values were imputed using mean substitution.

🧠 Machine Learning Models
The following models were trained and evaluated:

Logistic Regression

Random Forest Classifier

K-Nearest Neighbors (KNN)

Support Vector Classifier (SVC)

Each model was trained and evaluated using both the training and testing datasets to calculate accuracy.

📈 Evaluation Metric
Accuracy Score was used to evaluate model performance on both training and test datasets.

📌 How to Run
Open the .ipynb file in Google Colab.

Make sure your dataset is available in the specified Google Drive path:

swift
Copy
Edit
/content/drive/MyDrive/Projects/LoanApprovalPrediction.csv
Run each cell sequentially to mount the drive, preprocess data, and train models.

📷 Sample Visualizations
Bar plots of categorical variable distributions

Heatmap of feature correlations

Catplot showing the relationship between gender, marital status, and loan approval

✅ Results
Each classifier prints its accuracy on both training and testing sets. This helps determine the best-performing model for loan approval prediction.
