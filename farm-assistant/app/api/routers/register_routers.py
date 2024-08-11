from fastapi import FastAPI
from .home import router as home
from .analytics import router as analytics
from .auth import router as auth
from .chat import router as chat
from fastapi.staticfiles import StaticFiles


def register_routers(app: FastAPI) -> None:
    app.include_router(home)
    app.include_router(router=analytics)
    app.include_router(router=auth)
    app.include_router(router=chat)
    app.mount("/static", StaticFiles(directory="static"), name="static")