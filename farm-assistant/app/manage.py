from dotenv import load_dotenv
load_dotenv()
from api import create_app
from fastapi import FastAPI
import uvicorn


app: FastAPI = create_app()

if __name__ == '__main__':
    uvicorn.run(app=app)