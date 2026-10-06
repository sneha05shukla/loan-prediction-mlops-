import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
 
app = FastAPI(
    title = "Loan Prediction API",
    version="1.0"
)
 
model = joblib.load("./model/loan_default.pkl")
 
class LoanInput(BaseModel):
    current_loan_amount: float = Field(alias="Current Loan Amount")
    term: float = Field(alias="Term")
    credit_score: float = Field(alias="Credit Score")
    annual_income: float = Field(alias="Annual Income")
    years_in_current_job: float = Field(alias="Years in current job")
    home_ownership: float = Field(alias="Home Ownership")
    purpose: float = Field(alias="Purpose")
    monthly_debt: float = Field(alias="Monthly Debt")
    years_of_credit_history: float = Field(alias="Years of Credit History")
    months_since_last_delinquent: float = Field(alias="Months since last delinquent")
    number_of_open_accounts: float = Field(alias="Number of Open Accounts")
    number_of_credit_problems: float = Field(alias="Number of Credit Problems")
    current_credit_balance: float = Field(alias="Current Credit Balance")
    maximum_open_credit: float = Field(alias="Maximum Open Credit")
    bankruptcies: float = Field(alias="Bankruptcies")
    tax_liens: float = Field(alias="Tax Liens")
 
@app.get("/")
def home():
    return {
        "message": "Loan Prediction API"
    }
 
@app.post("/predict")
def predict(data: LoanInput):
    # FIX: Force Pydantic to use "Current Loan Amount" instead of "current_loan_amount"
    input_dict = data.model_dump(by_alias=True)
   
    # Convert the aliased dictionary into a pandas DataFrame
    input_data = pd.DataFrame([input_dict])
   
    # FIX: Ensure column positions perfectly line up with the training feature sequence
    input_data = input_data[model.feature_names_in_]
 
    # Run structural prediction
    prediction = model.predict(input_data)
 
    return {
        "prediction": float(prediction[0])
    }
 
 