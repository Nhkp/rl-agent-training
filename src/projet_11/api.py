from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from projet_11.agent import predict_action
from projet_11.settings import MODEL_PATH, OBSERVATION_SIZE


app = FastAPI(title="Eagle-1 LunarLander API")


class PlayRequest(BaseModel):
    observation: Annotated[
        list[float],
        Field(min_length=OBSERVATION_SIZE, max_length=OBSERVATION_SIZE),
    ]
    deterministic: bool = True


class PlayResponse(BaseModel):
    action: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/play", response_model=PlayResponse)
def play(request: PlayRequest):
    try:
        action = predict_action(request.observation, deterministic=request.deterministic)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PlayResponse(action=action)
