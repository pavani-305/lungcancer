from flask import Flask, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result_html = ""

    if request.method == "POST":
        features = [
            int(request.form["gender"]),
            int(request.form["age"]),
            int(request.form["smoking"]),
            int(request.form["yellow_fingers"]),
            int(request.form["anxiety"]),
            int(request.form["peer_pressure"]),
            int(request.form["chronic_disease"]),
            int(request.form["fatigue"]),
            int(request.form["allergy"]),
            int(request.form["wheezing"]),
            int(request.form["alcohol"]),
            int(request.form["coughing"]),
            int(request.form["shortness_of_breath"]),
            int(request.form["swallowing_difficulty"]),
            int(request.form["chest_pain"])
        ]

        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            result_html = "<div class='result danger'>⚠️ Lung Cancer Detected</div>"
        else:
            result_html = "<div class='result safe'>✅ No Lung Cancer Detected</div>"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lung Cancer Prediction</title>
        <style>
            body {{
                margin: 0;
                font-family: 'Segoe UI', sans-serif;
                background: #f4f7fb;
            }}

            .container {{
                max-width: 500px;
                margin: 40px auto;
                background: #ffffff;
                padding: 30px;
                border-radius: 14px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}

            h2 {{
                text-align: center;
                color: #1f2937;
                margin-bottom: 5px;
            }}

            p {{
                text-align: center;
                color: #6b7280;
                margin-bottom: 25px;
                font-size: 14px;
            }}

            label {{
                font-size: 13px;
                color: #374151;
                font-weight: 600;
            }}

            input {{
                width: 100%;
                padding: 10px;
                margin: 6px 0 15px 0;
                border-radius: 8px;
                border: 1px solid #d1d5db;
                font-size: 14px;
            }}

            input:focus {{
                outline: none;
                border-color: #2563eb;
            }}

            button {{
                width: 100%;
                padding: 12px;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.3s;
            }}

            button:hover {{
                background: #1e40af;
            }}

            .result {{
                margin-top: 20px;
                padding: 15px;
                text-align: center;
                border-radius: 10px;
                font-weight: bold;
                font-size: 16px;
            }}

            .danger {{
                background: #fee2e2;
                color: #b91c1c;
            }}

            .safe {{
                background: #dcfce7;
                color: #166534;
            }}

            footer {{
                text-align: center;
                margin-top: 25px;
                font-size: 12px;
                color: #9ca3af;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Lung Cancer Prediction</h2>
            <p>Enter patient details to predict risk</p>

            <form method="POST">
                <label>Gender (0 = Female, 1 = Male)</label>
                <input name="gender" required>

                <label>Age</label>
                <input name="age" required>

                <label>Smoking (0 / 1)</label>
                <input name="smoking" required>

                <label>Yellow Fingers (0 / 1)</label>
                <input name="yellow_fingers" required>

                <label>Anxiety (0 / 1)</label>
                <input name="anxiety" required>

                <label>Peer Pressure (0 / 1)</label>
                <input name="peer_pressure" required>

                <label>Chronic Disease (0 / 1)</label>
                <input name="chronic_disease" required>

                <label>Fatigue (0 / 1)</label>
                <input name="fatigue" required>

                <label>Allergy (0 / 1)</label>
                <input name="allergy" required>

                <label>Wheezing (0 / 1)</label>
                <input name="wheezing" required>

                <label>Alcohol (0 / 1)</label>
                <input name="alcohol" required>

                <label>Coughing (0 / 1)</label>
                <input name="coughing" required>

                <label>Shortness of Breath (0 / 1)</label>
                <input name="shortness_of_breath" required>

                <label>Swallowing Difficulty (0 / 1)</label>
                <input name="swallowing_difficulty" required>

                <label>Chest Pain (0 / 1)</label>
                <input name="chest_pain" required>

                <button type="submit">Predict Result</button>
            </form>

            {result_html}

            <footer>
                AI-based Lung Cancer Risk Prediction System
            </footer>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
