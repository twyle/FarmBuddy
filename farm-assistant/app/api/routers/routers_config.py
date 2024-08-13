from fastapi.templating import Jinja2Templates

TEMPLATES_DIR: str = "templates"
BOT_MESSAGE: str = """Hello there, how may I help you?
"""
INFECTED: str = (
    "Give me a brief description of {disease} and how to deal with it. If applicable mention "
    "specific pesticides and fungicides that can be used."
)
HEALTHY: str = (
    "Give me random advice on how to icrease the health level of my crops."
)
templates = Jinja2Templates(directory=TEMPLATES_DIR)