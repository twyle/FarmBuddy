from fastapi.templating import Jinja2Templates

TEMPLATES_DIR: str = "templates"
BOT_MESSAGE: str = """Hello there, how may I help you?
"""

templates = Jinja2Templates(directory=TEMPLATES_DIR)