from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from engine.stream import websocket_endpoint
from database.crud import get_recent_logs

app = FastAPI(title="Face Attendance API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    # On récupère les derniers pointages depuis Supabase pour alimenter le tableau de bord.
    # limit=50 pour avoir un historique un peu plus fourni que les 10 par défaut de crud.py ;
    # ajustez selon ce que vous voulez afficher.
    logs = get_recent_logs(limit=50)
    return templates.TemplateResponse(
        request=request, name="dashboard.html", context={"request": request, "logs": logs}
    )


@app.get("/scanner", response_class=HTMLResponse)
async def read_scanner(request: Request):
    return templates.TemplateResponse(request=request, name="scanner.html", context={"request": request})


@app.websocket("/ws/detect")
async def detect_faces(websocket: WebSocket):
    await websocket_endpoint(websocket)
