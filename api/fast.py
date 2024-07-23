<<<<<<< HEAD
<<<<<<< HEAD
# TODO: Import your package, replace this by explicit imports of what you need
from icu_watch_package.main import predict

=======
import pandas as pd
>>>>>>> master
from fastapi import FastAPI
=======
from fastapi import FastAPI, File, UploadFile, HTTPException
>>>>>>> master
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import tensorflow
from icu_watch_package.model_new import load_trained_model
from icu_watch_package.preprocessor import preprocess_input
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.state.model = load_trained_model()
model = load_trained_model()

@app.post("/predict")
def pred(file: UploadFile = File(...)):
    if model  is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    #try:
    if True:
        data = pd.read_csv(BytesIO(file.file.read()))
        print(data)
        input_data = preprocess_input(data)
        print(input_data)
        print(input_data.shape)
        print(model.summary())
        print('workin this stage')
        #prediction = app.state.model.predict(input_data)
        prediction = model.predict(input_data)


        print('workin this stage 1',prediction)
        prediction_classes = prediction #(prediction > 0.3).astype(int)
        print('workin this stage 2',prediction_classes)
        return {"predictions": prediction_classes.flatten().tolist()}
    #except Exception as e:
    #    raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"greeting": "Hello"}

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)
