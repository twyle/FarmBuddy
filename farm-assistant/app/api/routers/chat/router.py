from fastapi import Security, HTTPException, status
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from ..routers_config import templates, BOT_MESSAGE
from datetime import datetime
from pydantic import BaseModel
from ...extensions import ChatModel


class ChatMessage(BaseModel):
    role: str
    message: str
   

router = APIRouter(
    tags=["Chat"],
    prefix='/chat')


@router.get('/', status_code=status.HTTP_200_OK, response_class=HTMLResponse)
async def begin_chat(request: Request):
    """Load the home page"""
    time: str = str(datetime.now())
    return templates.TemplateResponse(
        "chat.html", 
        {
            "request": request,
            "bot_message": BOT_MESSAGE,
            "bot_time": time
        }
    )


@router.post('/', status_code=status.HTTP_201_CREATED)
async def chat_with_expert(request: Request, chat_message: ChatMessage):
    model: ChatModel = request.state.chat_model
    response: str = model.chat(message=chat_message.message)
    print(response)
    return ChatMessage(role='system', message=response)
