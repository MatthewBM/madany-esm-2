import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List

from inference import ESM2Engine
from config import settings

VALID_AMINO_ACIDS = re.compile(r"^[ACDEFGHIKLMNPQRSTVWXY]+$")

def validate_protein_sequence(v: str) -> str:
    """
    Validate that inputted protein sequences correspond to real amino acids.
    Force uppper-case usage and additionally check the length is under the config's max length.

    :param v: string of amino acids
    :return: validated and uppercased string
    """
    v = v.upper()
    if len(v) > settings.max_sequence_length:
        raise ValueError(f"Sequence length {len(v)} exceeds limit {settings.max_sequence_length}")
    if not VALID_AMINO_ACIDS.match(v):
        raise ValueError("Sequence contains invalid amino acid characters")
    return v

class SingleSequenceRequest(BaseModel):
    """
    Processing a single protein embedding request.
    Pydantic BaseModel will automatically handle JSON parsing and type checking.

    :param sequence: The amino acid string to be embedded
    """
    sequence: str = Field(..., min_length=1, description="Amino acid sequence")

    @field_validator("sequence")
    @classmethod
    def validate_single(cls, v: str) -> str:
        return validate_protein_sequence(v)

class BatchSequenceRequest(BaseModel):
    """
    Processing a multiple protein embedding requests.

    :param sequence: The amino acid string to be embedded
    """
    sequences: List[str] = Field(..., min_length=1, max_length=64, description="List of sequences")

    @field_validator("sequences")
    @classmethod
    def validate_batch(cls, v: List[str]) -> List[str]:
        return [validate_protein_sequence(seq) for seq in v]

"""
This global variable is the inference engine after it starts up.
Keep it at the module level so it persists across different web requests.
"""
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup/shutdown logic of entire API application.
    When the API starts, create ESM2Engine, spawning the separate worker processes and loading model weights
    When API stops, tell engine to cleanly shut down the worker pool

    :param app: The FastAPI application instance.
    """
    global engine
    engine = ESM2Engine()
    yield
    if engine and hasattr(engine, 'executor'):
        engine.executor.shutdown(wait=True)

app = FastAPI(title="ESM-2 Multi-GPU Inference API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    Create endpoint for Kubernetes probes and monitoring tools.
    Cause 503 if engine is not initialized
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine initializing...")

    return {
        "status": "healthy",
        "gpu_count_detected": engine.gpu_count,
        "is_mock_mode": engine.is_mock,
        "workers_active": engine.num_workers,
        "model_loaded": settings.model_name
    }

@app.post("/predict")
async def predict(request: SingleSequenceRequest):
    """
    Endpoint for requesting a single protein sequence's embeddings.
    Runs async so web server is still responsive while requests are processing.

    :param request: The validated SingleSequenceRequest object.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")
    try:
        embedding = await engine.predict_single(request.sequence)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchSequenceRequest):
    """
    Endpoint for requesting a multiple protein sequence's embeddings.
    Sends the sequences as list which inference engine which will split to seperate GPU-workers

    :param request: The validated BatchSequenceRequest object.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")
    try:
        embeddings = await engine.predict_batch(request.sequences)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    """
    Post webapp by wrapping FastAPI app in uvicorn web server.
    Binding 0.0.0.0 will causes security check fail,
    since this is containerized, security can be managed at kube Ingress/NetworkPolicy level, so adding #nosec
    """
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False) # nosec B104
