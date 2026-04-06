import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import shutil

# AI Upscaler
from realesrgan import RealESRGAN
import torch

app = FastAPI()

# Create folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load AI Model (once)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth')

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        input_path = f"{UPLOAD_FOLDER}/{file_id}.png"
        output_path = f"{OUTPUT_FOLDER}/{file_id}.png"

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Open image
        image = Image.open(input_path).convert("RGB")

        # Upscale image
        sr_image = model.predict(image)

        # Save output
        sr_image.save(output_path)

        # URL (Render will host static files)
        image_url = f"https://your-render-url.onrender.com/images/{file_id}.png"

        return JSONResponse({
            "success": True,
            "image_url": image_url
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })
