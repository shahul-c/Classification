from flask import Flask, render_template, request
import pickle
import numpy as np

# ------------------------------
# 1️⃣ Load model and scaler
# ------------------------------
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

# ------------------------------
# 2️⃣ Home route
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# ------------------------------
# 3️⃣ Prediction route
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        values = [float(x) for x in request.form.values()]

        # Convert to numpy array and reshape
        final_input = np.array(values).reshape(1, -1)

        # Scale input
        scaled_input = scaler.transform(final_input)

        # Predict wine type (0 = Red, 1 = White)
        prediction = model.predict(scaled_input)[0]

        # Interpret result
        if prediction == 0:
            result = "🍷 Red Wine"
        else:
            result = "🥂 White Wine"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# ------------------------------
# 4️⃣ Run Flask app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
