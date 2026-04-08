from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load model safely
try:
    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    print("✅ Model and encoder loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# ✅ Home route
@app.route("/")
def home():
    return "✅ Health AI API is running successfully!"

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # 🔍 Debug incoming data
        print("Received data:", data)

        # ✅ Input validation
        required_fields = [
            "fever", "cough", "fatigue", "difficulty_breathing",
            "age", "gender", "blood_pressure", "cholesterol"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # ✅ Convert inputs
        fever = 1 if data["fever"] == "Yes" else 0
        cough = 1 if data["cough"] == "Yes" else 0
        fatigue = 1 if data["fatigue"] == "Yes" else 0
        breathing = 1 if data["difficulty_breathing"] == "Yes" else 0

        age = int(data["age"])
        gender = 1 if data["gender"] == "Male" else 0

        bp_map = {"Low": 0, "Normal": 1, "High": 2}
        chol_map = {"Normal": 0, "High": 1}

        # ✅ Safe mapping (prevents crash)
        if data["blood_pressure"] not in bp_map:
            return jsonify({"error": "Invalid blood_pressure value"}), 400

        if data["cholesterol"] not in chol_map:
            return jsonify({"error": "Invalid cholesterol value"}), 400

        bp = bp_map[data["blood_pressure"]]
        cholesterol = chol_map[data["cholesterol"]]

        # ✅ Feature order MUST match training
        features = [[
            fever,
            cough,
            fatigue,
            breathing,
            age,
            gender,
            bp,
            cholesterol
        ]]

        # ✅ Prediction
        prediction = model.predict(features)
        disease = encoder.inverse_transform(prediction)[0]

        # ✅ Confidence (optional but recommended)
        confidence = max(model.predict_proba(features)[0]) * 100

        return jsonify({
            "predicted_disease": disease,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500


# ✅ Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)