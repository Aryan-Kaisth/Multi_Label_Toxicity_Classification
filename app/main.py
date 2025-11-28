from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from app.schema import ToxicityRequest
from src.pipelines.prediction_pipeline import PredictionPipeline

# FastAPI Initialization
app = FastAPI(title="Multi-Label Toxicity Classification")

# Static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML Templates
templates = Jinja2Templates(directory="templates")


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_text(payload: ToxicityRequest):
    """
    Accepts validated input text and returns toxicity predictions.
    Using your Pydantic model ensures:
    - min length validation
    - description metadata
    - example payloads in Swagger UI
    """
    try:
        predictor = PredictionPipeline()
        preds = predictor.predict(payload.text)
        return JSONResponse(preds)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
