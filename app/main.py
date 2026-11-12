"""CivicAI API: will a New York 311 request be resolved within 7 days?

Start it from the project root (so that the "src" package can be imported):

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/docs for the interactive documentation.
On a server use --host 0.0.0.0. Settings (all optional) are listed in app/predictor.py.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.predictor import ModelNotAvailable, Predictor
from app.schemas import (ErrorResponse, HealthResponse, PredictRequest, PredictResponse)

log = logging.getLogger("civicai")
if not log.handlers:                                  # show our own INFO and WARNING lines next to uvicorn's
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     [civicai] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

UNAVAILABLE = ("The prediction model is not loaded, so no prediction can be made. "
               "Please try again later or contact the service operator.")


# ---------------------------------------------------------------- readable 422 messages
def describe_problem(error, fields):
    """Turn one Pydantic error into (field name, plain-language message)."""
    kind = error["type"]
    location = [str(part) for part in error["loc"] if part != "body"]
    field = location[-1] if location else "body"
    inner = (error.get("ctx") or {}).get("error")

    if kind == "json_invalid":
        return "body", "The request body is not valid JSON. Check the commas, quotes and brackets."
    if kind in ("model_attributes_type", "dict_type"):
        return "body", "The request body must be one JSON object with the fields listed in /docs."
    if isinstance(inner, ValueError):                  # our own rules in app/schemas.py
        return getattr(inner, "field", field), str(inner)
    if kind == "missing":
        return field, f"{field} is required."
    if kind == "extra_forbidden":
        return field, f"{field} is not a field of this API. Check the spelling. The fields are: {', '.join(fields)}."
    if kind == "string_type":
        return field, f"{field} must be text (a string), for example \"BROOKLYN\"; numbers and other types are not accepted."
    if kind == "string_too_short":
        return field, f"{field} must not be empty. Leave the field out (or send null) if you have no value."
    if kind == "string_too_long":
        return field, f"{field} is too long ({error['ctx'].get('max_length')} characters at most)."
    if kind in ("float_type", "float_parsing"):
        return field, f"{field} must be a number, for example 40.69 (not text)."
    if kind in ("finite_number", "float_from_str"):
        return field, f"{field} must be a finite number."
    return field, f"{field}: {error['msg']}"


def error_body(error, message, problems=()):
    return ErrorResponse(error=error, message=message, errors=list(problems)).model_dump()


# ---------------------------------------------------------------- the app
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor().load()           # the only place where the model is read
    yield


def create_app():
    app = FastAPI(
        title="CivicAI",
        description="Predicts whether a New York City 311 service request will be resolved within 7 days. "
                    "Send one request to POST /predict. Class 0 (not resolved within 7 days) is the one staff want to flag.",
        version="2.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = list(PredictRequest.model_fields)
        problems = []
        for error in exc.errors():
            field, message = describe_problem(error, fields)
            problems.append({"field": field, "message": message})
        names = ", ".join(sorted({p["field"] for p in problems}))
        log.info("422 on %s: problem(s) with %s", request.url.path, names)          # field names only, never the values
        return JSONResponse(status_code=422, content=error_body(
            "validation_error", f"The request was not accepted. Fix the field(s) named below and send it again: {names}.", problems))

    @app.exception_handler(ModelNotAvailable)
    async def model_unavailable(request: Request, exc: ModelNotAvailable):
        return JSONResponse(status_code=503, content=error_body("model_unavailable", UNAVAILABLE))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        log.exception("Unexpected error on %s", request.url.path)                    # the details stay in the server log
        return JSONResponse(status_code=500, content=error_body(
            "internal_error", "Something went wrong on the server. The error has been logged."))

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}, tags=["service"])
    def health(request: Request):
        """200 when the model is loaded and ready, 503 when it is not."""
        predictor = request.app.state.predictor
        if not predictor.ready:
            return JSONResponse(status_code=503, content=HealthResponse(
                status="unavailable", model_loaded=False, message=UNAVAILABLE).model_dump())
        return HealthResponse(status="ok", model_loaded=True, model_version=str(predictor.meta["model_version"]),
                              versions_match_model_meta=not predictor.version_mismatches,
                              version_mismatches=predictor.version_mismatches)

    @app.get("/model-info", tags=["service"], responses={503: {"model": ErrorResponse}})
    def model_info(request: Request):
        """What the model is: version, input fields, label meaning, cut-off, split dates, training date and library versions."""
        return request.app.state.predictor.public_meta()

    @app.post("/predict", response_model=PredictResponse, tags=["prediction"],
              responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def predict(body: PredictRequest, request: Request):
        """Predict for one request. predicted_class is decided by the model's own cut-off, not by 0.5."""
        return request.app.state.predictor.predict(body)

    return app


app = create_app()
