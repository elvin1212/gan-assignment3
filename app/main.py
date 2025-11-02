# app/main.py
from fastapi import FastAPI, HTTPException, Query, Response
import matplotlib.pyplot as plt
import os

from .generator import GANGenerator

app = FastAPI(title="MNIST GAN Image Generator API")

MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/generator_best.pth")
generator = GANGenerator(MODEL_PATH)

@app.get("/")
def read_root():
    return {
        "message": "MNIST GAN Image Generator API",
        "endpoints": {
            "/": "This welcome message",
            "/generate": "GET - Generate new MNIST-like images",
            "/health": "GET - Health check",
            "/docs": "Interactive API documentation (Swagger UI)"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/generate")
def generate_images(
    response: Response,
    num_samples: int = Query(default=16, ge=1, le=64),
    format_type: str = Query(default="image/png", enum=["image/png", "json"])
):
    """
    Generate MNIST-like images using GAN.

    Query Parameters:
    - num_samples: Number of images to generate (1–64)
    - format_type: Response format (`image/png` or `json`)
        - `image/png`: returns a single PNG collage
        - `json`: returns list of Base64-encoded PNG strings
    """
    try:
        images = generator.generate_images(num_samples=num_samples)

        if format_type == "image/png":
            png_bytes = generator.images_to_png_bytes(images)
            return Response(content=png_bytes, media_type="image/png")

        elif format_type == "json":
            b64_images = generator.images_to_base64(images)
            return {"generated_images": b64_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
