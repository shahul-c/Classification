from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)

# ------------------------------
# 1️⃣ Load trained model
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_pipeline.pkl")

try:
    clf = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully from", MODEL_PATH)
except Exception as e:
    print("❌ Error loading model:", e)
    clf = None

# ------------------------------
# 2️⃣ Expected feature names
# ------------------------------
FEATURES = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area'
]

# ------------------------------
# 3️⃣ Routes
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if clf is None:
            return render_template("index.html", prediction="❌ Model not loaded properly!")

        # Read user input from form
        form = request.form.to_dict()
        print("📥 Received form data:", form)

        # Convert input fields
        row = {
            'Gender': form.get('Gender', 'Male'),
            'Married': form.get('Married', 'No'),
            'Dependents': form.get('Dependents', '0'),
            'Education': form.get('Education', 'Graduate'),
            'Self_Employed': form.get('Self_Employed', 'No'),
            'ApplicantIncome': float(form.get('ApplicantIncome', 0) or 0),
            'CoapplicantIncome': float(form.get('CoapplicantIncome', 0) or 0),
            'LoanAmount': float(form.get('LoanAmount', 0) or 0),
            'Loan_Amount_Term': float(form.get('Loan_Amount_Term', 0) or 0),
            'Credit_History': float(form.get('Credit_History', 1) or 1),
            'Property_Area': form.get('Property_Area', 'Urban')
        }

        # Ensure Dependents format
        if row['Dependents'] not in ['0', '1', '2', '3+']:
            row['Dependents'] = '0'

        print("🧾 Processed input row:", row)

        # Convert to DataFrame
        input_df = pd.DataFrame([row], columns=FEATURES)
        print("📊 Input DataFrame:\n", input_df)

        # --------------------------
        # Prediction
        # --------------------------
        pred = clf.predict(input_df)[0]
        print("✅ Prediction value:", pred)

        try:
            proba = clf.predict_proba(input_df)[0][1]
            print("📈 Prediction probability:", proba)
        except Exception as e:
            print("⚠️ Probability unavailable:", e)
            proba = None

        # Result message
        if pred == 1:
            message = "✅ Loan Approved"
        else:
            message = "❌ Loan Rejected"

        if proba is not None:
            message += f" (Confidence: {proba*100:.2f}%)"

        print("📢 Final message:", message)
        return render_template("index.html", prediction=message, last_input=row)

    except Exception as e:
        print("🔥 Error during prediction:", traceback.format_exc())
        return render_template("index.html", prediction=f"Error: {str(e)}")

# ------------------------------
# 4️⃣ Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
