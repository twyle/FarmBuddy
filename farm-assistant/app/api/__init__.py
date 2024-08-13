from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import register_routers
from .extensions import MaizeModel, ChatModel
from contextlib import asynccontextmanager
import os


origins = [
    "http://localhost",
    "http://localhost:8080",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model: MaizeModel = MaizeModel(model_path=os.environ['MODEL_DIRECTORY'])
    chat_model: ChatModel = ChatModel()
    yield {'model': model, 'chat_model': chat_model}
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