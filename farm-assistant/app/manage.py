from dotenv import load_dotenv
load_dotenv()
from api import create_app
from fastapi import FastAPI

app: FastAPI = create_app()