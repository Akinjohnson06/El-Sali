from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/Workspace/TDI/model.joblib")
le = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/Workspace/TDI/labelencoder.joblib")
scaler = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/Workspace/TDI/scaler.joblib")

# Define FastAPI app
app = FastAPI(title="Loan Approval Predictor")

# Define input schema using Pydantic
class LoanApplication(BaseModel):
    Age: int
    Gender: str
    Account_Type: str
    Account_Balance: float
    Transaction_Type: str
    Transaction_Amount: float
    Account_Age_Years: float
    Loan_Amount: float
    Loan_Type: str
    Interest_Rate: float
    Loan_Term: int
    Card_Type: str
    Credit_Limit: float
    Credit_Card_Balance: float
    Minimum_Payment_Due: float
    Rewards_Points: int
    Days_Since_Last_Transaction: int
    Days_Since_Last_Credit_Payment: int

@app.post("/predict")
def predict_loan_status(application: LoanApplication):
    # Convert input to dict and DataFrame
    data = application.dict()

    # Compute derived field
    if data['Transaction_Type'] == 'Deposit':
        data['Account_Balance_After_Transaction'] = data['Account_Balance'] + data['Transaction_Amount']
    else:
        data['Account_Balance_After_Transaction'] = data['Account_Balance'] - data['Transaction_Amount']

    # Add dummy Loan_Status (used in original feature list)
    data['Loan_Status'] = 0

    # Construct DataFrame
    df = pd.DataFrame([{
        'Age': data['Age'],
        'Gender': data['Gender'],
        'Account Type': data['Account_Type'],
        'Account Balance': data['Account_Balance'],
        'Transaction Type': data['Transaction_Type'],
        'Transaction Amount': data['Transaction_Amount'],
        'Account Balance After Transaction': data['Account_Balance_After_Transaction'],
        'Loan Amount': data['Loan_Amount'],
        'Loan Type': data['Loan_Type'],
        'Interest Rate': data['Interest_Rate'],
        'Loan Term': data['Loan_Term'],
        'Loan Status': data['Loan_Status'],
        'Card Type': data['Card_Type'],
        'Credit Limit': data['Credit_Limit'],
        'Credit Card Balance': data['Credit_Card_Balance'],
        'Minimum Payment Due': data['Minimum_Payment_Due'],
        'Rewards Points': data['Rewards_Points'],
        'Account Age (Years)': data['Account_Age_Years'],
        'Days Since Last Transaction': data['Days_Since_Last_Transaction'],
        'Days Since Last Credit Payment': data['Days_Since_Last_Credit_Payment']
    }])

    # Reorder columns
    columns_order = [
        'Age', 'Gender', 'Account Type', 'Account Balance', 'Transaction Type',
        'Transaction Amount', 'Account Balance After Transaction', 'Loan Amount',
        'Loan Type', 'Interest Rate', 'Loan Term', 'Loan Status', 'Card Type',
        'Credit Limit', 'Credit Card Balance', 'Minimum Payment Due', 'Rewards Points',
        'Account Age (Years)', 'Days Since Last Transaction', 'Days Since Last Credit Payment'
    ]
    df = df[columns_order]

    # Encode categorical variables
    df['Gender'] = le.transform(df['Gender'])
    df['Account Type'] = le.transform(df['Account Type'])
    df['Transaction Type'] = le.transform(df['Transaction Type'])
    df['Loan Type'] = le.transform(df['Loan Type'])
    df['Card Type'] = le.transform(df['Card Type'])


    # Scale numeric features
    columns_to_scale = [
        'Account Balance', 'Transaction Amount', 'Account Balance After Transaction',
        'Loan Amount', 'Interest Rate', 'Credit Limit', 'Credit Card Balance',
        'Minimum Payment Due', 'Account Age (Years)'
    ]
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    # Drop dummy
    df = df.drop('Loan Status', axis=1)

    # Make prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)

    result = {
        "prediction": int(prediction[0]),
        "probability_of_approval": round(probability[0][1], 4),
        "probability_of_rejection": round(probability[0][0], 4),
        "status": "Approved" if prediction[0] == 1 else "Rejected"
    }
    return result
