import os
import face_recognition
import pickle
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

KNOWN_FACES_DIR = 'static/images'
PICKLE_FILE = 'known_faces.pkl'
API_URL = 'https://project.pisofterp.com/pipl/restworld/employees'

EMPLOYEE_DATA = []

# Encode known faces
def encode_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
                print(f"[INFO] ✅ Face encoded for: {filename}")
            else:
                print(f"[WARNING] ❌ No face found in {filename}")

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
    print("[INFO] ✅ Encoding saved to known_faces.pkl")

# Load encodings
if not os.path.exists(PICKLE_FILE):
    encode_known_faces()

with open(PICKLE_FILE, 'rb') as f:
    data = pickle.load(f)

# Fetch employee data
def fetch_employee_data():
    global EMPLOYEE_DATA
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            EMPLOYEE_DATA = response.json()
            print("[INFO] ✅ Employee data fetched and cached.")
        else:
            print("[ERROR] ❌ Failed to fetch employee data.")
    except Exception as e:
        print(f"[ERROR] ❌ Exception while fetching employee data: {e}")

fetch_employee_data()

# HTML route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Recognition API
@app.post("/api/recognize")
async def recognize(request: Request):
    try:
        data_json = await request.json()
        if not data_json or 'image' not in data_json:
            raise HTTPException(status_code=400, detail='No image provided')

        image_data = data_json['image']
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        face_locations = face_recognition.face_locations(img_np)
        unknown_encodings = face_recognition.face_encodings(img_np, face_locations)

        if not unknown_encodings:
            raise HTTPException(status_code=400, detail='No face detected in uploaded image')

        unknown_encoding = unknown_encodings[0]
        face_distances = face_recognition.face_distance(data['encodings'], unknown_encoding)

        if len(face_distances) == 0:
            raise HTTPException(status_code=500, detail='No known faces available for comparison')

        best_match_index = np.argmin(face_distances)
        threshold = 0.5

        if face_distances[best_match_index] < threshold:
            matched_name = data['names'][best_match_index]
            print(f"[MATCH] Face matched with: {matched_name}")

            for emp in EMPLOYEE_DATA:
                if emp['employeeName'].lower() == matched_name.lower():
                    return {
                        'match': True,
                        'employeeName': emp['employeeName'],
                        'employeeId': emp['id'],
                        'encryptedId': emp.get('encryptedId', ''),
                        'image': emp['employeePic']
                    }
            raise HTTPException(status_code=404, detail='Matched name not found in API')
        else:
            raise HTTPException(status_code=404, detail='No face matched')

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)