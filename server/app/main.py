from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="JamTab API", version="0.1.0")


class ScaleSuggestion(BaseModel):
    section_id: str
    chord: str
    key_center: str
    recommended_scales: List[str]
    scale_degrees: List[str]
    target_notes: List[str]


class AnalysisResponse(BaseModel):
    chords: List[str]
    form: List[str]
    scale_suggestions: List[ScaleSuggestion]


class UsageResponse(BaseModel):
    is_subscribed: bool
    transcriptions_used: int
    free_quota: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/usage", response_model=UsageResponse)
def get_usage() -> UsageResponse:
    # TODO: replace with DB lookup and entitlement check
    return UsageResponse(is_subscribed=False, transcriptions_used=0, free_quota=5)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), instrument: Optional[str] = None, difficulty: Optional[str] = None) -> AnalysisResponse:
    # TODO: wire to AI pipeline. For now return stubbed analysis
    if file.content_type not in {"audio/mpeg", "audio/wav", "video/mp4", "audio/mp4", "audio/x-wav"}:
        raise HTTPException(status_code=400, detail="Unsupported content type")
    chords = ["G", "C", "D", "G"]
    form = ["A", "B", "A", "B"]
    scale_suggestions = [
        ScaleSuggestion(
            section_id="A",
            chord="G",
            key_center="G",
            recommended_scales=["G minor pentatonic", "G mixolydian"],
            scale_degrees=["1", "b3", "4", "5", "b7"],
            target_notes=["G", "D", "F"],
        )
    ]
    return AnalysisResponse(chords=chords, form=form, scale_suggestions=scale_suggestions)


