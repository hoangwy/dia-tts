# Dia TTS API Usage Guide

This guide describes how to use the REST API for the Dia TTS model.

## Getting Started

1. **Start the API Server:**
   ```bash
   ./docker-helper.sh api
   ```

2. **Access Documentation:**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Endpoints

### 1. Health Check
Check if the API is running and the model is loaded.

- **URL:** `/api/health`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_name": "nari-labs/Dia-1.6B-0626"
  }
  ```

### 2. Generate Speech (TTS)
Generate audio from text.

- **URL:** `/api/tts/generate`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "text": "[S1] Hello world! [S2] This is a test.",
    "output_format": "mp3",
    "guidance_scale": 3.0,
    "temperature": 1.8
  }
  ```
- **Parameters:**
  - `text` (required): Input text with speaker tags `[S1]`, `[S2]`.
  - `output_format`: `mp3` (default), `wav`, or `base64`.
  - `guidance_scale`: 1.0 - 10.0 (default 3.0).
  - `temperature`: 0.1 - 2.0 (default 1.8).
  - `seed`: Optional integer for reproducibility.

### 3. Voice Cloning
Generate speech using a cloned voice from an uploaded audio file.

- **URL:** `/api/tts/generate-with-voice`
- **Method:** `POST` (Multipart Form Data)
- **Form Fields:**
  - `text`: Text to generate.
  - `transcript`: Transcript of the uploaded audio.
  - `audio_file`: The audio file to clone (WAV/MP3).

## Client Examples

### Python Client

```python
import requests

API_URL = "http://localhost:8000/api/tts/generate"

payload = {
    "text": "[S1] Welcome to the Dia TTS API.",
    "output_format": "mp3"
}

response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    with open("output.mp3", "wb") as f:
        f.write(response.content)
    print("Audio saved to output.mp3")
else:
    print("Error:", response.text)
```

### cURL

```bash
curl -X POST http://163.5.212.63:27623/api/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "On December 3rd, The Early Signing Period is officially underway, and the Louisville Cardinals football program is buzzing with activity. Coaches Jeff Brohm and Vince Marrow have been working tirelessly, securing over twenty verbal commitments. Today is the day these promising young athletes make their decisions official by signing their letters of intent. This period is always a whirlwind of emotions for fans and coaches alike. While some players might decommit and choose another school, there's also the excitement of recruits flipping their commitments to join the Cardinals. It's a dynamic day filled with constant updates, and we're here to keep you informed every step of the way. Among the notable early signees are Jarvis Strickland, a four-star offensive tackle from Paducah, Kentucky, and Julius Miles, another four-star talent playing wide receiver from Freeport, Florida. These top recruits are expected to make a significant impact on the team in the coming seasons. Additionally, the Cardinals have secured commitments from several other promising players. These include Taj Powell, a three-star linebacker from Ohio, and Sam Dawson, a three-star defensive lineman from Crestwood, Kentucky. The roster is shaping up with a strong mix of talent across various positions. As the day progresses, we'll continue to monitor all the signings and provide updates on every player who officially joins the Louisville Cardinals during this crucial Early Signing Period.", "output_format": "mp3"}' \
  --output curl_test.mp3
```

### Base64 Response

To get JSON with base64 encoded audio:

```bash
curl -X POST http://localhost:8000/api/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "[S1] Base64 test.", "output_format": "base64"}'
```
