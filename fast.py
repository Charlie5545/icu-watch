import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    # TODO: Implement actual model loading logic
    return None

def preprocess_features(data):
    # TODO: Implement actual preprocessing logic
    return data

# Preload the model
app.state.model = load_model()

@app.get("/predict")
def predict(
        Hour: int,
        HR: float,
        O2Sat: float,
        Temp: float,
        SBP: float,
        MAP: float,
        DBP: float,
        Resp: float,
        EtCO2: float,
        Age: float,
        Gender: str
    ):
    """
    Make a single sepsis prediction.
    """
    data = {
        "Hour": Hour,
        "HR": HR,
        "O2Sat": O2Sat,
        "Temp": Temp,
        "SBP": SBP,
        "MAP": MAP,
        "DBP": DBP,
        "Resp": Resp,
        "EtCO2": EtCO2,
        "Age": Age,
        "Gender_Female": 1 if Gender == 'Female' else 0,
        "Gender_Male": 1 if Gender == 'Male' else 0
    }

    X_pred = pd.DataFrame([data])

    model = app.state.model
    if model is not None:
        X_processed = preprocess_features(X_pred)
        y_pred = model.predict(X_processed)
        sepsis_prediction = int(y_pred[0])
    else:
        # Dummy prediction if model is not available
        sepsis_prediction = 0

    return {"sepsis": sepsis_prediction}

@app.get("/")
def root():
    return {"greeting": "Hello"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
