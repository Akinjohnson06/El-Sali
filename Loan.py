import streamlit as st
import pandas as pd
import joblib

# Load the saved models and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/Workspace/TDI/model.joblib")
    le = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/Workspace/TDI/labelencoder.joblib")
    scaler = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/Workspace/TDI/scaler.joblib")
    return model, le, scaler

model, le, scaler = load_artifacts()

# Define the categorical columns and their options
gender_options = ['Male', 'Female', 'Other']
account_type_options = ['Current', 'Savings']
transaction_type_options = ['Withdrawal', 'Deposit', 'Transfer']
loan_type_options = ['Mortgage', 'Auto', 'Personal']
card_type_options = ['AMEX', 'MasterCard', 'Visa']

# Streamlit app
def main():
    st.title("Loan Status Prediction App")
    st.write("This app predicts whether a loan application will be approved or rejected based on customer financial data.")

    with st.form("loan_prediction_form"):
        st.header("Customer Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", gender_options)
            account_type = st.selectbox("Account Type", account_type_options)
            account_balance = st.number_input("Account Balance", min_value=0.0, value=5000.0)

        with col2:
            transaction_type = st.selectbox("Transaction Type", transaction_type_options)
            transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
            account_age = st.number_input("Account Age (Years)", min_value=0.0, max_value=50.0, value=5.0)

        st.header("Loan Information")
        col3, col4 = st.columns(2)

        with col3:
            loan_amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0)
            loan_type = st.selectbox("Loan Type", loan_type_options)
            interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=20.0, value=5.0)
            loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=120, value=36)

        with col4:
            card_type = st.selectbox("Card Type", card_type_options)
            credit_limit = st.number_input("Credit Limit", min_value=0.0, value=3000.0)
            credit_card_balance = st.number_input("Credit Card Balance", min_value=0.0, value=1000.0)
            min_payment_due = st.number_input("Minimum Payment Due", min_value=0.0, value=50.0)

        st.header("Additional Information")
        col5, col6 = st.columns(2)

        with col5:
            rewards_points = st.number_input("Rewards Points", min_value=0, value=5000)
            days_last_transaction = st.number_input("Days Since Last Transaction", min_value=0, value=30)

        with col6:
            days_last_credit_payment = st.number_input("Days Since Last Credit Payment", min_value=0, value=30)

        submitted = st.form_submit_button("Predict Loan Approval")

    if submitted:
        input_data = {
            'Age': age,
            'Gender': gender,
            'Account Type': account_type,
            'Account Balance': account_balance,
            'Transaction Type': transaction_type,
            'Transaction Amount': transaction_amount,
            'Account Balance After Transaction': account_balance + transaction_amount if transaction_type == 'Deposit' else account_balance - transaction_amount,
            'Loan Amount': loan_amount,
            'Loan Type': loan_type,
            'Interest Rate': interest_rate,
            'Loan Term': loan_term,
            'Loan Status': 0,  # dummy
            'Card Type': card_type,
            'Credit Limit': credit_limit,
            'Credit Card Balance': credit_card_balance,
            'Minimum Payment Due': min_payment_due,
            'Rewards Points': rewards_points,
            'Account Age (Years)': account_age,
            'Days Since Last Transaction': days_last_transaction,
            'Days Since Last Credit Payment': days_last_credit_payment
        }

        columns_order = [
            'Age', 'Gender', 'Account Type', 'Account Balance', 'Transaction Type',
            'Transaction Amount', 'Account Balance After Transaction', 'Loan Amount',
            'Loan Type', 'Interest Rate', 'Loan Term', 'Loan Status', 'Card Type',
            'Credit Limit', 'Credit Card Balance', 'Minimum Payment Due', 'Rewards Points',
            'Account Age (Years)', 'Days Since Last Transaction', 'Days Since Last Credit Payment'
        ]

        input_df = pd.DataFrame([input_data])[columns_order]

        # Encode categorical variables
        input_df['Gender'] = le.fit_transform(input_df['Gender'])
        input_df['Account Type'] = le.fit_transform(input_df['Account Type'])
        input_df['Transaction Type'] = le.fit_transform(input_df['Transaction Type'])
        input_df['Loan Type'] = le.fit_transform(input_df['Loan Type'])
        input_df['Card Type'] = le.fit_transform(input_df['Card Type'])

        # Scale numerical features
        columns_to_scale = [
            'Account Balance', 'Transaction Amount', 'Account Balance After Transaction',
            'Loan Amount', 'Interest Rate', 'Credit Limit', 'Credit Card Balance',
            'Minimum Payment Due', 'Account Age (Years)'
        ]
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        # Drop Loan Status
        input_df = input_df.drop('Loan Status', axis=1)

        # Predict
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Results:")
        if prediction[0] == 1:
            st.success("Loan Status: Approve ✅")
            st.write(f"Probability of Approval: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.error("Loan Status: Reject ❌")
            st.write(f"Probability of Rejection: {prediction_proba[0][0]*100:.2f}%")

        
if __name__ == '__main__':
    main()
