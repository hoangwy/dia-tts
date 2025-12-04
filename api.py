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
import time
import numpy as np

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
    
    # Record start time for generation
    generation_start_time = time.time()
    
    try:
        # Set seed if provided
        import torch
        if request.seed is not None:
            torch.manual_seed(request.seed)
            
        # Text chunking logic
        import re
        # Split by sentence endings (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', request.text)
        chunks = []
        current_chunk = ""
        max_chars = 400  # Conservative limit to avoid context window issues
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        if not chunks:
            chunks = [request.text]
            
        logger.info(f"Split text into {len(chunks)} chunks for generation")
        
        audio_segments = []
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            logger.info(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            # Generate audio for chunk
            segment = model.generate(
                chunk,
                cfg_scale=request.guidance_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_new_tokens,
                cfg_filter_top_k=request.top_k
            )
            
            chunk_elapsed_time = time.time() - chunk_start_time
            logger.info(f"Chunk {i+1}/{len(chunks)} generated in {chunk_elapsed_time:.2f} seconds")
            
            # Ensure segment is a numpy array
            if not isinstance(segment, np.ndarray):
                # If it's a tensor, convert to numpy
                if hasattr(segment, 'cpu'):
                    segment = segment.cpu().numpy()
                else:
                    segment = np.array(segment)
            
            # Ensure 2D array (channels, samples)
            if segment.ndim == 1:
                segment = segment[np.newaxis, :]
                
            audio_segments.append(segment)
            logger.debug(f"Chunk {i+1} shape: {segment.shape}, dtype: {segment.dtype}")
            
        # Concatenate audio segments
        try:
            if len(audio_segments) > 1:
                logger.info(f"Concatenating {len(audio_segments)} audio segments...")
                # Ensure all segments have the same shape except for the time dimension
                # Get the reference shape from the first segment
                ref_shape = audio_segments[0].shape
                logger.debug(f"Reference shape: {ref_shape}")
                
                # Verify all segments have compatible shapes
                for i, seg in enumerate(audio_segments):
                    if seg.shape[:-1] != ref_shape[:-1]:
                        logger.warning(f"Segment {i} shape {seg.shape} doesn't match reference {ref_shape}")
                        # Reshape to match if needed
                        if seg.ndim != len(ref_shape):
                            if seg.ndim == 1 and len(ref_shape) == 2:
                                seg = seg[np.newaxis, :]
                            elif seg.ndim == 2 and len(ref_shape) == 1:
                                seg = seg[0] if seg.shape[0] == 1 else seg
                        # Ensure channel dimension matches
                        if seg.shape[0] != ref_shape[0] and ref_shape[0] == 1:
                            seg = seg[:1] if seg.shape[0] > 1 else seg
                        audio_segments[i] = seg
                
                # Concatenate along the last axis (time)
                full_audio = np.concatenate(audio_segments, axis=-1)
                logger.info(f"Concatenation successful. Final shape: {full_audio.shape}")
            else:
                full_audio = audio_segments[0]
                logger.info(f"Single segment, no concatenation needed. Shape: {full_audio.shape}")
        except Exception as e:
            logger.error(f"Error during audio concatenation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error concatenating audio segments: {str(e)}")
        
        # Validate audio before saving
        if full_audio is None or full_audio.size == 0:
            logger.error("Generated audio is empty or None")
            raise HTTPException(status_code=500, detail="Generated audio is empty")
        
        logger.info(f"Audio validation: shape={full_audio.shape}, dtype={full_audio.dtype}, size={full_audio.size}")
        
        # CRITICAL: Ensure audio is always 2D numpy array with shape (samples, channels)
        # soundfile.write() requires shape[1] to exist for channels
        # Convert to numpy if it's not already
        if not isinstance(full_audio, np.ndarray):
            import torch
            if isinstance(full_audio, torch.Tensor):
                full_audio = full_audio.cpu().numpy()
                logger.info(f"Converted torch tensor to numpy: {full_audio.shape}")
            else:
                full_audio = np.array(full_audio)
                logger.info(f"Converted to numpy array: {full_audio.shape}")
        
        # Ensure 2D shape: (samples, channels)
        if full_audio.ndim == 1:
            # 1D array -> (samples, 1)
            full_audio = full_audio[:, np.newaxis]
            logger.info(f"Converted 1D to 2D: {full_audio.shape}")
        elif full_audio.ndim == 2:
            # Check if we need to transpose from (channels, samples) to (samples, channels)
            dim0, dim1 = full_audio.shape
            # If first dimension is small (1-2) and second is much larger, likely (channels, samples)
            if dim0 <= 2 and dim1 > dim0 * 100:
                full_audio = full_audio.T
                logger.info(f"Transposed from ({dim0}, {dim1}) to {full_audio.shape}")
            # Verify final shape makes sense: samples should be much larger than channels
            final_samples, final_channels = full_audio.shape
            if final_samples < final_channels:
                # Still wrong, transpose again
                full_audio = full_audio.T
                logger.warning(f"Shape still wrong, transposed again to {full_audio.shape}")
        else:
            # More than 2D - reshape to 2D
            logger.warning(f"Audio has {full_audio.ndim} dimensions, reshaping to 2D")
            # Flatten all but the last dimension, or reshape to (total_samples, 1)
            if full_audio.size > 0:
                full_audio = full_audio.reshape(-1, 1)
                logger.info(f"Reshaped to: {full_audio.shape}")
        
        # Final check: must be 2D with shape (samples, channels) where samples >> channels
        if full_audio.ndim != 2:
            logger.error(f"CRITICAL: Audio is not 2D! Shape: {full_audio.shape}, ndim: {full_audio.ndim}")
            full_audio = full_audio.reshape(-1, 1)
            logger.warning(f"Force reshaped to: {full_audio.shape}")
        
        samples, channels = full_audio.shape
        if samples < channels:
            logger.error(f"CRITICAL: samples ({samples}) < channels ({channels}) - transposing!")
            full_audio = full_audio.T
            samples, channels = full_audio.shape
        
        logger.info(f"Final audio shape: {full_audio.shape} (samples={samples}, channels={channels})")
        
        # Use numpy array directly - don't convert to torch tensor
        # model.save_audio() should handle numpy arrays correctly
        audio_for_save = full_audio
        
        # Save to temporary file
        try:
            logger.info(f"Saving audio to temporary file (format: {request.output_format})...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.output_format}") as tmp_file:
                tmp_path = tmp_file.name
            # Save outside the context manager to ensure file is closed
            logger.debug(f"Calling model.save_audio with shape {full_audio.shape if hasattr(full_audio, 'shape') else type(full_audio)}")
            model.save_audio(audio_for_save, tmp_path)
            logger.info(f"Audio saved successfully to {tmp_path}")
        except Exception as e:
            logger.error(f"Error saving audio file: {e}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Log the shape that caused the error
            if hasattr(audio_for_save, 'shape'):
                logger.error(f"Audio shape that caused error: {audio_for_save.shape}, ndim: {len(audio_for_save.shape) if hasattr(audio_for_save, 'shape') else 'N/A'}")
            # Try to fix and retry if it's a shape issue
            if "tuple index out of range" in str(e) or "shape" in str(e).lower():
                logger.info("Attempting to fix shape and retry...")
                try:
                    # Ensure it's a numpy array with correct shape
                    if not isinstance(audio_for_save, np.ndarray):
                        import torch
                        if isinstance(audio_for_save, torch.Tensor):
                            audio_for_save = audio_for_save.cpu().numpy()
                    
                    # Force 2D shape
                    if audio_for_save.ndim == 1:
                        audio_for_save = audio_for_save[:, np.newaxis]
                    elif audio_for_save.ndim == 2:
                        s, c = audio_for_save.shape
                        if s < c:
                            audio_for_save = audio_for_save.T
                    
                    logger.info(f"Retrying with fixed shape: {audio_for_save.shape}")
                    model.save_audio(audio_for_save, tmp_path)
                    logger.info(f"Audio saved successfully after shape fix to {tmp_path}")
                except Exception as e2:
                    logger.error(f"Error saving after shape fix: {e2}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Error saving audio: {str(e2)}")
            else:
                raise HTTPException(status_code=500, detail=f"Error saving audio: {str(e)}")
        
        # Calculate and log total generation time
        total_elapsed_time = time.time() - generation_start_time
        logger.info(f"TTS generation completed in {total_elapsed_time:.2f} seconds (total time)")
        
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
