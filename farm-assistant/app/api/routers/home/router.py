from fastapi import Security, HTTPException, status
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from ..routers_config import templates
   

router = APIRouter(
    tags=["Home"],)




@router.get('/', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def get_home_page(request: Request):
    """Load the home page"""
    return templates.TemplateResponse("home.html", {"request": request})


@router.get('/dashboard', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def get_dahsboard(request: Request):
    """Manage tokens"""
    return templates.TemplateResponse("dashboard.html", {"request": request})
