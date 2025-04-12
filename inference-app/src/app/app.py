"""FastAPI server for the model inference."""

# mypy: disable-error-code="import-not-found,misc"

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root() -> dict[str, str]:
    """Check if the server is running."""
    return {"status": "ready"}


@app.post("/get_embeddings")
def get_embeddings(sequences: list[str]) -> list[float]:
    """Get embeddings for a list of sequences.

    Args:
        sequences: list of sequences to get embeddings for

    Returns:
        list of embeddings
    """
    return [0.0] * len(sequences)


@app.post("/run_inference")
def run_inference(sequences: list[str]) -> list[str]:
    """Run inference for a list of sequences.

    Args:
        sequences: list of sequences to run inference for

    Returns:
        list of outputs
    """
    outputs = []

    for sequence in sequences:
        outputs.append(sequence)

    return outputs
