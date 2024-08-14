from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import register_routers
from .extensions import MaizeModel, ChatModel, PestModel
from contextlib import asynccontextmanager
import os


origins = [
    "http://localhost",
    "http://localhost:8080",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    maize_model: MaizeModel = MaizeModel(model_path=os.environ['MAIZE_MODEL_DIRECTORY'])
    pest_model: PestModel = PestModel(model_path=os.environ['PEST_MODEL_DIRECTORY'])
    chat_model: ChatModel = ChatModel()
    yield {'maize_model': maize_model, 'chat_model': chat_model, 'pest_model': pest_model}
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