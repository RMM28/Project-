import os

import shutil

from fastapi import FastAPI, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware

import whisper
 
# Set path for ffmpeg if required by Whisper

os.environ["PATH"] += os.pathsep + r"C:\\ffmpeg\\ffmpeg-7.1.1-essentials_build\bin"
 
app = FastAPI()
 
# CORS settings to allow frontend access (adjust in production)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],  # Use actual frontend domain in prod

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)
 
# Load Whisper model only once at startup for efficiency

model = whisper.load_model("base")
 
@app.post("/transcribe")

async def transcribe_audio(file: UploadFile = File(...)):

    try:

        # Save uploaded audio temporarily

        temp_path = "temp_audio.wav"

        with open(temp_path, "wb") as buffer:

            shutil.copyfileobj(file.file, buffer)
 
        # Transcribe using Whisper

        result = model.transcribe(temp_path)

        transcript = result.get("text", "")
 
        return {"transcript": transcript}

    except Exception as e:

        return {"error": str(e)}

    finally:

        # Clean up temporary file

        if os.path.exists(temp_path):

            os.remove(temp_path)

 