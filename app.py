from toxicpred.pipeline.train_pipeline import TrainPipeline
from fastapi import FastAPI, File, UploadFile,Body
from starlette.responses import RedirectResponse
from starlette import status
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from toxicpred.pipeline.prediction_pipeline import PredictionPipeline
from toxicpred.constant.application import APP_HOST, APP_PORT
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_routed():
    try:

        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(csv_file: UploadFile = File(...)):
    try:
       
        df = pd.read_csv(csv_file.file)
        prediction_pipeline = PredictionPipeline()
        #validate the data
        valid_status = prediction_pipeline.validate(df)
        if not valid_status:
            return Response("Invalid input")
        predictions = prediction_pipeline.predict(df)
        if not predictions:
            return Response("Model is not available")
        return { "prediction": predictions}
        
    except Exception as e:
        raise Response(f"Error Occured! {e}")


#if __name__ == "__main__":
#    app_run(app, host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

