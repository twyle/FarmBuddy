from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import register_routers
from .extensions import (
    MLModel, load_maize_model, load_model, MAIZE_LABELS, PEST_LABELS, TOMATO_LABELS, ChatModel
)
from contextlib import asynccontextmanager
import os


origins = [
    "http://localhost",
    "http://localhost:8080",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    maize_model: MLModel = MLModel(
        model_path=os.environ['MAIZE_MODEL_DIRECTORY'],
        model_loader=load_maize_model,
        labels=MAIZE_LABELS,
    )
    pest_model: MLModel = MLModel(
        model_path=os.environ['PEST_MODEL_DIRECTORY'],
        model_loader=load_model,
        labels=PEST_LABELS,
    )
    tomato_model: MLModel = MLModel(
        model_path=os.environ['TOMATO_MODEL_DIRECTORY'],
        model_loader=load_model,
        labels=TOMATO_LABELS,
    )
    
    chat_model: ChatModel = ChatModel()
    yield {
        'maize_model': maize_model, 
        'chat_model': chat_model, 
        'pest_model': pest_model, 
        'tomato_model': tomato_model, 
    }
    # Clean up the ML models and release the resources


def create_app():
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routers(app=app)
    
    return app