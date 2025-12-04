"""
FastAPI server for Dia TTS model
Provides REST API endpoints for text-to-speech generation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import io
import base64
import tempfile
import os
from pathlib import Path
import logging
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dia TTS API",
    description="Text-to-Speech API using nari-labs/dia model for ultra-realistic dialogue generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech. Must include speaker tags like [S1] and [S2]")
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0, description="Guidance scale for generation quality (higher = more guidance)")
    temperature: float = Field(1.8, ge=0.1, le=2.0, description="Sampling temperature (higher = more variation)")
    top_p: float = Field(0.90, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(45, ge=0, le=100, description="Top-k sampling parameter")
    max_new_tokens: int = Field(3072, ge=256, le=8192, description="Maximum number of tokens to generate")
    output_format: Literal["mp3", "wav", "base64"] = Field("mp3", description="Output format for the audio")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "[S1] Hello, this is a test. [S2] Wow, that sounds amazing!",
                "guidance_scale": 3.0,
                "temperature": 1.8,
                "output_format": "mp3"
            }
        }


class VoiceCloneRequest(BaseModel):
    text: str = Field(..., description="Text to generate with cloned voice")
    transcript: str = Field(..., description="Transcript of the uploaded audio (must include speaker tags)")
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0)
    temperature: float = Field(1.8, ge=0.1, le=2.0)
    output_format: Literal["mp3", "wav", "base64"] = Field("mp3")


class TTSResponse(BaseModel):
    status: str
    message: str
    audio_base64: Optional[str] = None
    format: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    checkpoint: str
    loaded: bool


# Startup event to load model
@app.on_event("startup")
async def load_model():
    """Load the Dia TTS model on startup"""
    global model
    try:
        logger.info("Loading Dia TTS model...")
        from dia import Dia
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626")
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - redirect to docs"""
    return {
        "message": "Dia TTS API Server",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name="nari-labs/Dia-1.6B-0626" if model is not None else None
    )


@app.get("/api/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models():
    """List available models"""
    return [
        ModelInfo(
            name="Dia 1.6B",
            checkpoint="nari-labs/Dia-1.6B-0626",
            loaded=model is not None
        )
    ]


@app.post("/api/tts/generate", tags=["TTS"])
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text using Dia TTS model
    
    Returns audio file in the specified format (mp3, wav, or base64)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        # Set seed if provided
        import torch
        if request.seed is not None:
            torch.manual_seed(request.seed)
        
        # Generate audio
        audio = model.generate(
            request.text,
            guidance_scale=request.guidance_scale,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.output_format}") as tmp_file:
            tmp_path = tmp_file.name
            model.save_audio(audio, tmp_path)
        
        # Handle different output formats
        if request.output_format == "base64":
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_path)
            audio_b64 = base64.b64encode(audio_bytes).decode()
            return TTSResponse(
                status="success",
                message="Audio generated successfully",
                audio_base64=audio_b64,
                format=request.output_format
            )
        else:
            # Return file response
            media_type = "audio/mpeg" if request.output_format == "mp3" else "audio/wav"
            return FileResponse(
                tmp_path,
                media_type=media_type,
                filename=f"generated.{request.output_format}",
                background=None  # File will be cleaned up after response
            )
    
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


@app.post("/api/tts/generate-with-voice", tags=["TTS"])
async def generate_with_voice_clone(
    text: str,
    transcript: str,
    audio_file: UploadFile = File(...),
    guidance_scale: float = 3.0,
    temperature: float = 1.8,
    output_format: Literal["mp3", "wav"] = "mp3"
):
    """
    Generate speech with voice cloning from uploaded audio
    
    Upload an audio file with its transcript, and the model will generate
    new speech matching the voice in the uploaded audio.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating speech with voice cloning...")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            content = await audio_file.read()
            tmp_audio.write(content)
            audio_path = tmp_audio.name
        
        # Generate with voice cloning
        import torch
        import torchaudio
        
        # Load audio prompt
        audio_prompt, sr = torchaudio.load(audio_path)
        
        # Combine transcript and generation text
        full_text = f"{transcript} {text}"
        
        audio = model.generate(
            full_text,
            audio_prompt=audio_prompt,
            guidance_scale=guidance_scale,
            temperature=temperature
        )
        
        # Save output
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp_out:
            out_path = tmp_out.name
            model.save_audio(audio, out_path)
        
        # Cleanup input file
        os.unlink(audio_path)
        
        # Return response
        media_type = "audio/mpeg" if output_format == "mp3" else "audio/wav"
        return FileResponse(
            out_path,
            media_type=media_type,
            filename=f"cloned.{output_format}"
        )
    
    except Exception as e:
        logger.error(f"Error in voice cloning: {e}")
        raise HTTPException(status_code=500, detail=f"Error in voice cloning: {str(e)}")


@app.post("/api/tts/batch", tags=["TTS"])
async def batch_generate(requests: list[TTSRequest]):
    """
    Generate multiple audio files in batch
    
    Note: This processes sequentially. For parallel processing, make separate requests.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")
    
    results = []
    for idx, req in enumerate(requests):
        try:
            logger.info(f"Processing batch request {idx + 1}/{len(requests)}")
            
            audio = model.generate(
                req.text,
                guidance_scale=req.guidance_scale,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_new_tokens=req.max_new_tokens
            )
            
            # Save to output directory with index
            output_path = OUTPUT_DIR / f"batch_{idx}.{req.output_format}"
            model.save_audio(audio, str(output_path))
            
            results.append({
                "index": idx,
                "status": "success",
                "output_file": str(output_path)
            })
        except Exception as e:
            results.append({
                "index": idx,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
