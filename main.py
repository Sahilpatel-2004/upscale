from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
import cv2

app = FastAPI()

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Serve images
app.mount("/images", StaticFiles(directory=OUTPUT_FOLDER), name="images")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        input_path = f"{UPLOAD_FOLDER}/{file_id}.png"
        output_path = f"{OUTPUT_FOLDER}/{file_id}.png"

        # Save file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read image
        img = cv2.imread(input_path)

        # Upscale (2x)
        upscaled = cv2.resize(
            img,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # Save output
        cv2.imwrite(output_path, upscaled)

        image_url = f"https://upscale-1-7tsq.onrender.com/images/{file_id}.png"

        return JSONResponse({
            "success": True,
            "image_url": image_url
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })
