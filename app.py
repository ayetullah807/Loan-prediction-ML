import gradio as gr
import pandas as pd
import pickle


# Load trained model
with open("best_loan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("loan_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def predict_loan_status(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    property_area
):
    # Create input dataframe with exact training column names
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # Predict
    prediction = model.predict(input_df)
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]

    # Make result more user-friendly
    if decoded_prediction == "Y":
        return "Loan Approved"
    else:
        return "Loan Not Approved"


# Gradio UI
app = gr.Interface(
    fn=predict_loan_status,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Yes", "No"], label="Married"),
        gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education"),
        gr.Radio(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Applicant Income", value=5000),
        gr.Number(label="Coapplicant Income", value=0),
        gr.Number(label="Loan Amount", value=120),
        gr.Number(label="Loan Amount Term", value=360),
        gr.Radio([1.0, 0.0], label="Credit History"),
        gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property Area")
    ],
    outputs=gr.Text(label="Prediction"),
    title="Loan Approval Prediction System",
    description="Enter applicant details to predict whether the loan will be approved or not."
)

app.launch()