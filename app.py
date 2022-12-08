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
import jwt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.hash import bcrypt
from controller.model import PostSchema, UserSchema, UserLoginSchema
from controller.auth.auth_bearer import JWTBearer
from controller.auth.auth_handler import signJWT
app = FastAPI()

users = []
def check_user(data: UserLoginSchema):
    for user in users:
        if user.email == data.email and user.password == data.password:
            return True
    return False


@app.get("/train", dependencies=[Depends(JWTBearer())])
async def train_routed():
    try:

        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict", dependencies=[Depends(JWTBearer())])
async def predict_route(csv_file: UploadFile = File(...)):
    try:
       
        df = pd.read_csv(csv_file.file)
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(df)
        if not predictions:
            return Response("Model is not available")
        return { "prediction": predictions}
        
    except Exception as e:
        raise Response(f"Error Occured! {e}")


@app.post("/user/signup", tags=["user"])
def create_user(user: UserSchema = Body(...)):
    users.append(user) # replace with db call, making sure to hash the password first
    return signJWT(user.email)


@app.post("/user/login", tags=["user"])
def user_login(user: UserLoginSchema = Body(...)):
    if check_user(user):
        return signJWT(user.email)
    return {
        "error": "Wrong login details!"
    }

#if __name__ == "__main__":
 #   app_run(app, host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

