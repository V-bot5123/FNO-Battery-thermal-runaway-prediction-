from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from fno_model import FNO2d
import uvicorn
import os

app = FastAPI(
    title="FNO Battery Thermal Prediction API",
    description="Real-time API predicting battery thermal runaway using Fourier Neural Operators."
)

# Load the trained model globally when the API starts
MODEL_PATH = '../models/fno_battery_model.pt'
model = FNO2d(modes=12, width=32)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
else:
    print("Warning: Model file not found. Ensure you run train.py first.")

# Define the input schema
class SensorData(BaseModel):
    # Expecting a flat list of 4096 values (64x64 grid) or a 2D list
    grid: list[list[float]] 

@app.get("/")
def read_root():
    return {"message": "Battery Thermal FNO API is running. Send POST requests to /predict"}

@app.post("/predict")
def predict_thermal_state(data: SensorData):
    try:
        # Convert JSON input to a PyTorch tensor
        # Input shape expected: 64x64
        input_tensor = torch.tensor(data.grid, dtype=torch.float32)
        
        # FNO expects batch dimension: shape (1, 64, 64)
        input_tensor = input_tensor.unsqueeze(0)
        
        # Run inference instantly
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # Convert the predicted tensor back to a Python list
        output_grid = prediction.squeeze(0).tolist()
        
        return {
            "status": "success",
            "message": "Prediction generated instantly.",
            "predicted_heat_grid": output_grid
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    print("Starting API on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
