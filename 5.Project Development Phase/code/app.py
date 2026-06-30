import os
import pickle
import numpy as np

from flask import Flask, jsonify, render_template, request

from data.crop_info import crop_info

app = Flask(__name__)

MODEL_PATH = "models/crop_model.pkl"
SCALER_PATH = "models/scaler.pkl"
INFO_PATH = "models/model_info.pkl"

  
# Load Files
  

if (
    os.path.exists(MODEL_PATH)
    and os.path.exists(SCALER_PATH)
    and os.path.exists(INFO_PATH)
):

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(INFO_PATH, "rb") as f:
        model_info = pickle.load(f)

    print("Model Loaded Successfully")
    print(model_info)

else:

    model = None
    scaler = None
    model_info = None

    print("Model files not found.")


  
# Home
  

@app.route("/")
def home():
    return render_template("index.html")


  
# Prediction
  

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        return jsonify(
            {
                "success": False,
                "message": "Model not found."
            }
        )

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

        if model_info["requires_scaling"]:

            prediction_input = scaler.transform(features)

        else:

            prediction_input = features

        prediction = model.predict(prediction_input)[0]

        probabilities = model.predict_proba(prediction_input)[0]

        confidence = round(np.max(probabilities) * 100, 2)

        info = crop_info.get(
            prediction.lower(),
            {
                "emoji": "🌱",
                "water": "N/A",
                "temperature": "N/A",
                "description": "No information available."
            }
        )

        print("\nPrediction :", prediction)
        print("Confidence :", confidence)

        return jsonify(
            {
                "success": True,
                "crop": prediction,
                "confidence": confidence,
                "info": info
            }
        )

    except Exception as e:

        return jsonify(
            {
                "success": False,
                "message": str(e)
            }
        )


  
# Run App
  

if __name__ == "__main__":
    app.run(debug=True)