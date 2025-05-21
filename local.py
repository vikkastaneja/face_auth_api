# client_local_capture.py
# Captures webcam images, sends them to prediction API, and saves them locally

import cv2
import requests
import tempfile
import os
from PIL import Image
from datetime import datetime
import shutil

# URL of the FastAPI prediction endpoint
API_URL = "http://localhost:8000/predict/vote_batch" 

# Number of frames to capture
FRAME_COUNT = 5
FRAME_INTERVAL = 5  # Capture every N frames to space out images

# Output directory to save captured images
REVIEW_DIR = os.path.abspath("reviewed_frames")
os.makedirs(REVIEW_DIR, exist_ok=True)

print(f"📁 Saving review images to: {REVIEW_DIR}")

# Temporary directory to store frames for API
TEMP_DIR = tempfile.mkdtemp(prefix="face_auth_tmp_")
print(f"📁 Capturing frames to: {TEMP_DIR}")

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit(1)

frame_id = 0
saved = 0
while saved < FRAME_COUNT:
    ret, frame = cap.read()
    if not ret:
        print(f"⚠️ Failed to read frame {frame_id}")
        continue

    if frame_id % FRAME_INTERVAL == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"capture_{saved}_{timestamp}.jpg"
        temp_path = os.path.join(TEMP_DIR, fname)
        review_path = os.path.join(REVIEW_DIR, fname)

        # Save in temp for sending
        cv2.imwrite(temp_path, frame)
        # Save for review locally
        cv2.imwrite(review_path, frame)

        print(f"✅ Frame saved to: {review_path}")
        saved += 1

    frame_id += 1

cap.release()

# Prepare files for POST
files = [("files", (os.path.basename(p), open(os.path.join(TEMP_DIR, p), "rb"), "image/jpeg"))
         for p in sorted(os.listdir(TEMP_DIR))]

print("🚀 Sending frames to prediction API...")
try:
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        print("✅ Authentication Result:")
        print(response.json())
    else:
        print(f"❌ API Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"❌ Request failed: {e}")

# Cleanup temp
shutil.rmtree(TEMP_DIR)
print(f"🧹 Cleaned up temp files from {TEMP_DIR}")
