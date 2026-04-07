from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
import cv2
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.mount("/images", StaticFiles(directory=OUTPUT_FOLDER), name="images")


@app.get("/")
def home():
    return {"message": "🚀 High Quality Upscale API Running"}


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

        if img is None:
            return JSONResponse({"success": False, "error": "Invalid image"})

        # 🔥 STEP 1: Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # 🔥 STEP 2: Upscale 4x (High Quality)
        upscaled = cv2.resize(
            img,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_LANCZOS4
        )

        # 🔥 STEP 3: Sharpening
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(upscaled, -1, kernel)

        # 🔥 STEP 4: Contrast Enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Save
        cv2.imwrite(output_path, final_img)

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
