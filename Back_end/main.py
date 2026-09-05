from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import joblib

app = FastAPI(title="Smart Crop Recommendation API")

model = keras.models.load_model("best_crop_mlp.keras")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/")
def home():
    return {"message": "Smart Crop Recommendation API is running"}

@app.post("/predict")
def predict_crop(data: CropInput):
    input_data = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]]

    input_scaled = scaler.transform(input_data)

    probabilities = model.predict(input_scaled, verbose=0)[0]

    sorted_indices = probabilities.argsort()[::-1]

    threshold = 0.01

    predictions = []

    for index in sorted_indices:
        confidence = probabilities[index]

        if confidence >= threshold:
            crop = label_encoder.inverse_transform([index])[0]

            predictions.append({
                "crop": crop,
                "confidence": round(float(confidence * 100), 2)
            })

    return {
        "predictions": predictions
    }