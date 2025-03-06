from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request structure
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features).tolist()
    return {"predict": prediction}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
