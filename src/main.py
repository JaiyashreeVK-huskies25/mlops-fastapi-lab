from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import BaseModel, Field

from predict import MODEL_PATH, predict

app = FastAPI(title="Wine Classifier API")


class WineData(BaseModel):
    alcohol: float = Field(..., examples=[13.2])
    malic_acid: float = Field(..., examples=[1.78])
    ash: float = Field(..., examples=[2.14])
    alcalinity_of_ash: float = Field(..., examples=[11.2])
    magnesium: float = Field(..., examples=[100.0])
    total_phenols: float = Field(..., examples=[2.65])
    flavanoids: float = Field(..., examples=[2.76])
    nonflavanoid_phenols: float = Field(..., examples=[0.26])
    proanthocyanins: float = Field(..., examples=[1.28])
    color_intensity: float = Field(..., examples=[4.38])
    hue: float = Field(..., examples=[1.05])
    od280_od315: float = Field(..., examples=[3.40])
    proline: float = Field(..., examples=[1050.0])


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: Dict[str, float]


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict_wine(data: WineData):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run train.py first.")
    features = list(data.model_dump().values())
    return predict(features)
