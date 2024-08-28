from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = pickle.load(open('heart_clf.pkl', 'rb'))

# Define a request body model
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def read_root():
    return {'message': "Heart Desease Prediction API"}

# Define the prediction endpoint
@app.post("/predict/")
def predict_heart_disease(input_data: HeartDiseaseInput):
    data = [
        [
            input_data.age, input_data.sex, input_data.cp, input_data.trestbps,
            input_data.chol, input_data.fbs, input_data.restecg, input_data.thalach,
            input_data.exang, input_data.oldpeak, input_data.slope, input_data.ca, 
            input_data.thal
        ]
    ]
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    
    if prediction[0] == 0:
        result = "Heart Disease was not Detected"
    else:
        result = "Heart Disease was Detected"
    
    return {
        "prediction": result,
        "probability": prediction_proba[0].tolist()
    }
