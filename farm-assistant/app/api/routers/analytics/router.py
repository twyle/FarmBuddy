from fastapi import status, HTTPException
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from ..routers_config import templates
from ...extensions import MaizeModel
from typing import Annotated
from PIL import Image
import uuid
from os import path
import pathlib
   

router = APIRouter(
    tags=["Analytics"],
    prefix='/analysis')

def save_image(file: UploadFile) -> str:
    extension: str = pathlib.Path(file.filename).suffix
    image = Image.open(file.file)
    file_name: str = f"{str(uuid.uuid4())}{extension}"
    files_path: str = "/home/lyle/Professional Projects/model-deployments/farm-assistant/app/static/files"
    file_path: str = path.join(files_path, file_name)
    image.save(file_path) 
    return file_name


@router.post('/analyze', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def analyze_image(request: Request, file: Annotated[UploadFile, File]):
    """Manage tokens"""
    # if file.content_type not in ['image/gif']:
    #     return HTTPException(status_code=400, detail="Only images of type jpeg and png are accepted!")
    model: MaizeModel = request.state.model
    analysis: dict = model.analyze_image(image=Image.open(file.file))
    print(analysis)
    file_uri: str = save_image(file=file)
    print(file_uri)
    return templates.TemplateResponse(
        "analysis.html", 
        {
            "request": request,
            "analysis": analysis
        }
    )
    

@router.get('/', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def analyze(request: Request):
    return templates.TemplateResponse(
        "analysis.html", 
        {
            "request": request
        }
    )
    
    
@router.get('/upload', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def upload_image(request: Request):
    return templates.TemplateResponse(
        "popup.html", 
        {
            "request": request
        }
    )