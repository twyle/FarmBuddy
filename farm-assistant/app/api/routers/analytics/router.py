from fastapi import status, HTTPException
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from ..routers_config import templates, INFECTED, HEALTHY
from ...extensions import MaizeModel, ChatModel, PestModel, TomatoModel
from typing import Annotated
from PIL import Image
import uuid
from os import path
import pathlib
from .utils import geocode_location, get_agrovets, get_aggrovet_details
import os
   

router = APIRouter(
    tags=["Analytics"],
    prefix='/analysis')

session: dict = {}
distances: dict = {
    'immediate': 100,
    'near': 500,
    'far': 1000
}

def save_image(file: UploadFile) -> str:
    extension: str = pathlib.Path(file.filename).suffix
    image = Image.open(file.file)
    file_name: str = f"{str(uuid.uuid4())}{extension}"
    files_path: str = "/home/lyle/Professional Projects/model-deployments/farm-assistant/app/static/files"
    file_path: str = path.join(files_path, file_name)
    image.save(file_path) 
    return file_name


@router.post('/analyze', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def analyze_image(request: Request, file: Annotated[UploadFile, File], model: Annotated[str, Form()]):
    """Manage tokens"""
    # if file.content_type not in ['image/gif']:
    #     return HTTPException(status_code=400, detail="Only images of type jpeg and png are accepted!")
    print(model)
    if model == 'maize':
        maize_model: MaizeModel = request.state.maize_model
        analysis: dict = maize_model.evaluate_image(image=Image.open(file.file))
    elif model == 'tomato':
        tomato_model: TomatoModel = request.state.tomato_model
        analysis: dict = tomato_model.evaluate_image(image=Image.open(file.file))
    else:
        pest_model: PestModel = request.state.pest_model
        analysis: dict = pest_model.evaluate_image(image=Image.open(file.file))
    print(analysis)
    disease: str = analysis['prediction']
    response: str = None
    print(disease)
    chat_model: ChatModel = request.state.chat_model
    if disease == 'Healthy':
        response = chat_model.chat(message=HEALTHY)
    else:
        query: str = INFECTED.format(disease=disease)
        response = chat_model.chat(message=query)
    file_uri: str = save_image(file=file)
    print(file_uri)
    print(response)
    return templates.TemplateResponse(
        "analysis.html", 
        {
            "request": request,
            "analysis": analysis,
            "disease": response,
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
    

@router.get('/find_aggrovets', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def find_aggrovets(request: Request):
    return templates.TemplateResponse(
        "pick_aggrovets.html", 
        {
            "request": request
        }
    )
    
    
@router.post('/display_aggrovets', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def display_aggrovets(
    request: Request, 
    location: Annotated[str, Form()],
    radius: Annotated[str, Form()]
    ):
    api_key: str = os.environ['GOOGLE_MAPS_API_KEY']
    center: dict = geocode_location(location=location) 
    aggrovets: list[dict[str, float]] = get_agrovets(location=center['address'])
    print(aggrovets)
    context: dict = {
        'api_key': api_key,
        'zoom': 15,
        'map_id': 'FARM_BUDDY',
        'center': center['location'],
        'aggrovets': aggrovets
    }
    return templates.TemplateResponse(
        "aggrovets.html", 
        {
            "request": request,
            "context": context
        }
    )
    
@router.get('/location', status_code=status.HTTP_200_OK)
def get_user_location():
    location_str: str = session['location']
    location: dict[str, float] = geocode_location(location=location_str)
    return location