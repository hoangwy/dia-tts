# Dia TTS - Docker Setup

Docker configuration for running [nari-labs/dia](https://github.com/nari-labs/dia), a TTS model capable of generating ultra-realistic dialogue.

## Prerequisites

- **Docker**: Install [Docker Engine](https://docs.docker.com/engine/install/)
- **NVIDIA GPU**: Required for optimal performance
- **NVIDIA Container Toolkit**: Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for GPU support

### Verify GPU Support

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Build the Docker Image

```bash
docker-compose build
```

Or build directly:

```bash
docker build -t dia-tts:latest .
```

### Run the Container

#### Using Docker Compose (Recommended)

```bash
# Run the default example
docker-compose up

# Run the Gradio web interface
docker-compose run --rm --service-ports dia-tts python app.py

# Run the FastAPI server
docker-compose run --rm --service-ports dia-tts uvicorn api:app --host 0.0.0.0 --port 8000

# Run the CLI
docker-compose run --rm dia-tts python cli.py --help
```

#### Using Docker directly

```bash
# Run simple example
docker run --rm --gpus all dia-tts:latest

# Run Gradio app
docker run --rm --gpus all -p 7860:7860 dia-tts:latest python app.py

# Run API server
docker run --rm --gpus all -p 8000:8000 dia-tts:latest uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Usage

The project includes a FastAPI server for REST API access.

```bash
# Start the API
./docker-helper.sh api
```

- **Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health
- **Generate:** POST http://localhost:8000/api/tts/generate

See [API-USAGE.md](API-USAGE.md) for detailed documentation.

## Usage Examples

### 1. Simple Text-to-Speech

```bash
docker-compose run --rm dia-tts python -c "
from dia import Dia

model = Dia.from_pretrained('nari-labs/Dia-1.6B-0626')
text = '[S1] Hello, this is a test. [S2] Wow, that sounds great!'
audio = model.generate(text)
model.save_audio(audio, 'outputs/test.mp3')
"
```

### 2. Run Examples

```bash
# Simple example
docker-compose run --rm dia-tts python example/simple.py

# Voice cloning example
docker-compose run --rm dia-tts python example/voice_clone.py
```

### 3. Interactive Development

```bash
# Start interactive shell
docker-compose run --rm dia-tts bash

# Inside container, you can:
python example/simple.py
python app.py
python cli.py --help
```

## GPU Configuration

### Use Specific GPU

```bash
# In docker-compose.yml, modify:
environment:
  - CUDA_VISIBLE_DEVICES=0  # Use GPU 0 only
```

### Multiple GPUs

```bash
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1
```

## Volume Mounts

- `./outputs:/app/outputs` - Generated audio files
- `model-cache:/root/.cache/huggingface` - Cached model weights

## Troubleshooting

### GPU Not Detected

```bash
# Check if nvidia-container-toolkit is installed
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

Reduce batch size or use a smaller model. The default model requires significant VRAM.

### Slow First Run

The first run downloads model weights (~3GB). Subsequent runs use the cached models.

## CPU-Only Mode (Not Recommended)

For CPU-only inference (slower):

1. Modify `Dockerfile` to use a non-CUDA base image:
```dockerfile
FROM ubuntu:22.04
```

2. Remove GPU deployment in `docker-compose.yml`:
```yaml
# Remove the deploy section
```

## Advanced Configuration

### Custom Model

Modify the example scripts to use different model checkpoints:

```python
model = Dia.from_pretrained('nari-labs/Dia-YourModel')
```

### Environment Variables

Create a `.env` file:

```bash
CUDA_VISIBLE_DEVICES=0
MODEL_CHECKPOINT=nari-labs/Dia-1.6B-0626
```

## Resources

- [Dia GitHub Repository](https://github.com/nari-labs/dia)
- [Hugging Face Space](https://huggingface.co/nari-labs)
- [Discord Community](https://discord.gg/bJq6vjRRKv)

## License

This Docker setup follows the Apache License 2.0 of the original dia project.
