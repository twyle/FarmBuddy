from fastapi import Security, HTTPException, status
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from ..routers_config import templates
   

router = APIRouter(
    tags=["Auth"],
    prefix='/auth')




@router.get('/', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def profile(request: Request):
    """Load the home page"""
    return templates.TemplateResponse("profile.html", {"request": request})
