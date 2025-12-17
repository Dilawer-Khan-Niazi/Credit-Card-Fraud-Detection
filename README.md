\# Credit Card Fraud Detection Project

\## Overview

Hi! The main objective of this project involves creating a machine learning model which detects credit card fraud.

The training of my predictive model relied on a credit card transaction database to determine fraudulent from legitimate transactions.

The data set required special handling due to its low fraud case rates at 0.17% therefore I needed to select the model carefully.

The Random Forest model I selected used my set threshold of 0.3 to provide suitable fraud detection capabilities without creating unnecessary false reports of non-fraud activity.

The provided ZIP file contains all necessary materials to understand the project description alongside operation instructions for the model.

\## Contents of the ZIP File

Visionary Data Systems has prepared the following contents which exist inside the archive file.

The \*\*credit\_card\_fraud\_report.pdf\*\* contains complete documentation that explains the project journey from data loading to model training to deployment planning.

The document includes visual displays of results such as class distributions and feature importance alongside plans regarding further development.

Therefore I used the Jupyter Notebook named 'credit\_card\_fraud\_pipeline.ipynb' that contained all my code for building the pipeline. The project process included steps to load data then preprocess it and balance it before selecting features for model training followed by evaluation testing and model saving.

The trained Random Forest model for fraud detection uses the name credit\_card\_fraud\_model\_rf.pkl.

The StandardScaler I applied to preprocess the \`Amount\` column is stored as the file \*\*scaler.pkl\*\*.

This file contains the fifteen most important features (V14, V10, etc.) that the prediction model exploits.

The model determines transaction fraud status using the threshold value of 0.3 that resides within optimal\_threshold.pkl.

\## How to Use the Files

If you want to use the model to predict fraud on new transactions, here’s what you need to do:

\### 1. Set Up Your Environment

\- Make sure you have Python installed on your computer.

\- Install the required libraries by running this command in your terminal:

pip install pandas numpy scikit-learn imblearn xgboost matplotlib seaborn joblib

\- You’ll also need Jupyter Notebook to open the \`.ipynb\` file if you want to run the code.

\### 2. Run the Code (Optional)

\- Open \`credit\_card\_fraud\_pipeline.ipynb\` in Jupyter Notebook.

\- Update the file path in the code to point to your copy of the \`creditcard.csv\` dataset. For example:

\`\`\`python

file\_path = 'path/to/your/creditcard.csv'

3\. Use the Model for Predictions

If you just want to use the saved model to make predictions on new transactions, you can use the following Python script. This script loads the model and other files, preprocesses a new transaction, and predicts if it’s fraud.

python

Collapse

Wrap

Copy

import pandas as pd

import joblib

from datetime import datetime

\# Load the saved model and preprocessing objects

model = joblib.load('credit\_card\_fraud\_model\_rf.pkl')

scaler = joblib.load('scaler.pkl')

selected\_features = joblib.load('selected\_features.pkl')

optimal\_threshold = joblib.load('optimal\_threshold.pkl')

\# Reference date for Time conversion

reference\_date = pd.to\_datetime('2013-01-01 00:00:00')

\# Function to preprocess a new transaction

def preprocess\_transaction(data):

\# Convert the transaction to a DataFrame

df = pd.DataFrame(\[data\])

\# Convert Time to Hour

df\['Time'\] = reference\_date + pd.to\_timedelta(df\['Time'\], unit='s')

df\['Hour'\] = df\['Time'\].dt.hour

df = df.drop(columns=\['Time'\])

\# Scale the Amount column

df\['Amount'\] = scaler.transform(df\[\['Amount'\]\])

\# Select the top 15 features

X = df\[selected\_features\]

return X

\# Example transaction (replace with your own data)

new\_transaction = {

"Time": 0,

"V1": -1.3598071336738,

"V2": -0.0727811733098497,

"V3": 2.53634673796914,

"V4": 1.37815522427443,

"V5": -0.338320769942518,

"V6": 0.462387777762292,

"V7": 0.239598554205154,

"V8": 0.0986979012610507,

"V9": 0.363786969611213,

"V10": 0.0907941719789316,

"V11": -0.551599533260813,

"V12": -0.617800855762348,

"V13": -0.991389847235408,

"V14": -0.311169353699879,

"V15": 1.46817697209427,

"V16": -0.470400525259478,

"V17": 0.207971241929242,

"V18": 0.0257905801985591,

"V19": 0.403992960255733,

"V20": 0.251412098239705,

"V21": -0.018306777944153,

"V22": 0.277837575558899,

"V23": -0.110473910188767,

"V24": 0.0669280749146731,

"V25": 0.128539358273528,

"V26": -0.189114843888824,

"V27": 0.133558376740387,

"V28": -0.0210530534538215,

"Amount": 149.62

}

\# Preprocess the transaction

X\_new = preprocess\_transaction(new\_transaction)

\# Predict fraud probability

fraud\_prob = model.predict\_proba(X\_new)\[:, 1\]\[0\]

\# Apply the threshold to classify

prediction = 1 if fraud\_prob >= optimal\_threshold else 0

\# Print the result

print(f"Prediction: {prediction} (1 = Fraud, 0 = Non-Fraud)")

print(f"Fraud Probability: {fraud\_prob:.4f}")

Save this script as predict\_fraud.py (or any name you like).

Update the file paths in the script to point to the .pkl files in your directory.

Run the script to predict if the transaction is fraud.

The example transaction above should return a prediction of 0 (non-fraud) with a low fraud probability.

4\. Deploy the Model (Optional)

The document contains a complete guidance for model execution (see "Model Deployment Plan" section).

The document explains how to surround the model with API frameworks including Flask or FastAPI before making real-time predictions as well as showing

performance surveillance methods and continuous retraining procedures. You can use the given deployment plan to put your model into productive use if

desired.

Project Details

The applied model used Random Forest with a threshold value of 0.3.

Performance on Test Set:

The model identifies 86 out of 98 actual fraud cases with a recall rate of 0.8776.

The precision rate stands at 0.5850 demonstrating that 58 percent of the cases identified as fraudulent are indeed fraudulent.

F1-Score: 0.7020

AUC-ROC: 0.9731

Why Threshold 0.3? I picked this threshold that would identify more fraudulent cases without generating an excessive number of incorrect flags.

The default threshold of 0.5 achieved a higher F1-score yet passed over fraud instances that I wanted detected.
