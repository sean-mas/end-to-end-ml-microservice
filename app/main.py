"""FastAPI service: predicts NYC taxi trip duration with the production model.

The model is loaded once at startup, not per request. If MLflow can't be reached
the API still starts, /health reports why, and /predict answers 503 until a
restart finds the model.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from trip_duration import config

from . import model as model_module
from .schemas import BatchPrediction, Health, Prediction, TripRequest

logger = logging.getLogger(__name__)
MAX_BATCH = 1_000  # trips per batch request


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.service = None
    app.state.load_error = None
    try:
        app.state.service = model_module.load_service()
    except Exception as error:  # any failure is reported on /health
        logger.exception("Could not load the production model")
        app.state.load_error = f"{type(error).__name__}: {error}"
    yield


app = FastAPI(
    title="Trip Duration API",
    description=(
        "Predicts how many minutes a New York City yellow taxi trip will take, "
        "from pickup zone, dropoff zone and pickup time, all known at booking. "
        "Serves the model version marked `production` in the MLflow registry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _service(request: Request) -> model_module.ModelService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model loaded: {request.app.state.load_error}",
        )
    return service


@app.get("/health", response_model=Health, tags=["health"])
def health(request: Request) -> JSONResponse:
    """Say whether a model is loaded and which registry version it is."""
    service = request.app.state.service
    body = Health(
        status="ok" if service else "unavailable",
        model_name=config.setting("REGISTERED_MODEL_NAME"),
        model_alias=config.setting("MODEL_ALIAS"),
        model_version=service.version if service else None,
        detail=None if service else request.app.state.load_error,
    )
    return JSONResponse(body.model_dump(), status_code=200 if service else 503)


@app.post("/predict", response_model=Prediction, tags=["predictions"])
def predict(trip: TripRequest, request: Request) -> Prediction:
    """Predict the duration of one trip in minutes."""
    service = _service(request)
    [minutes] = service.predict([trip])
    return Prediction(
        predicted_duration_minutes=minutes,
        model_name=service.name,
        model_version=service.version,
    )


@app.post("/predict_batch", response_model=BatchPrediction, tags=["predictions"])
def predict_batch(trips: list[TripRequest], request: Request) -> BatchPrediction:
    """Predict several trips at once; the answers keep the request order."""
    if len(trips) > MAX_BATCH:
        raise HTTPException(422, f"At most {MAX_BATCH} trips per request.")
    service = _service(request)
    minutes = service.predict(trips) if trips else []
    return BatchPrediction(
        predicted_duration_minutes=minutes,
        model_name=service.name,
        model_version=service.version,
    )
