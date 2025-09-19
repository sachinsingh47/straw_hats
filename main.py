# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # or pickle, keras.models, torch.load depending on your model

# Define input schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # add all required features here

# Load model
model = joblib.load("model.pkl")  # replace with your saved model

# Create app
app = FastAPI(title="ML Model API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "ML Model API is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input into format expected by model
    features = [[
        data.feature1,
        data.feature2,
        data.feature3
        # add more in same order as training
    ]]
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
