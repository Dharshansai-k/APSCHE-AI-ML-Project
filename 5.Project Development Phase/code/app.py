import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from data.crop_info import crop_info

app = Flask(__name__)

MODEL_PATH = "models/crop_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print("Model and Scaler loaded successfully!")

else:
    model = None
    scaler = None
    print("ERROR: Model files not found. Please run train.py first.")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None:
        return jsonify({
            "success": False,
            "message": "Model not loaded."
        }), 500

    try:

        data = request.get_json()

        features = np.array([[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]])

        # Scale the input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        probabilities = model.predict_proba(features_scaled)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        info = crop_info.get(prediction.lower(), {
        "emoji": "🌱",
        "water": "N/A",
        "temperature": "N/A",
        "description": "No information available."
         })

        return jsonify({
            "success": True,
            "crop": prediction,
            "confidence": confidence,
            "info": info
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500



if __name__ == "__main__":
    app.run(debug=True)