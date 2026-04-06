from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
import cv2

app = FastAPI()

# ✅ Enable CORS (important for Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ✅ Serve output images
app.mount("/images", StaticFiles(directory=OUTPUT_FOLDER), name="images")


# ✅ Root route (fix 404 issue)
@app.get("/")
def home():
    return {"message": "🚀 Image Upscale API is running"}


# ✅ Upload & Upscale API
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 🔍 Validate file type
        if not file.content_type.startswith("image/"):
            return JSONResponse({
                "success": False,
                "error": "Only image files are allowed"
            })

        file_id = str(uuid.uuid4())

        input_path = f"{UPLOAD_FOLDER}/{file_id}.png"
        output_path = f"{OUTPUT_FOLDER}/{file_id}.png"

        # 💾 Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 🖼 Read image safely
        img = cv2.imread(input_path)

        if img is None:
            return JSONResponse({
                "success": False,
                "error": "Invalid image file"
            })

        # 🚀 Upscale (2x → can change to 4x)
        upscaled = cv2.resize(
            img,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # 💾 Save output
        cv2.imwrite(output_path, upscaled)

        # 🔗 Return correct Render URL
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
