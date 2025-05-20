# face_auth_api/main.py

from fastapi import FastAPI, File, UploadFile, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import torch
import joblib
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from skimage.feature import hog
from io import BytesIO
import os
import cv2

app = FastAPI(title="Face Authentication API")

# ---------- Load Models ----------
svm_model_path = "models/hog_svm_best.pkl"
cnn_model_path = "models/best_cnn.pth"
facenet_ref_path = "models/facenet_reference.pkl"

if not all(os.path.exists(p) for p in [svm_model_path, cnn_model_path, facenet_ref_path]):
    raise FileNotFoundError("One or more model files are missing. Please ensure all models are pre-generated.")

svm = joblib.load(svm_model_path)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)

cnn_model = CNN()
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
cnn_model.eval()

facenet = InceptionResnetV1(pretrained="vggface2").eval()
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)
facenet_data = joblib.load(facenet_ref_path)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def read_image(file: UploadFile):
    return Image.open(BytesIO(file.file.read())).convert("RGB")

@app.post("/predict/vote_batch")
def predict_vote_batch(files: List[UploadFile] = File(...)):
    svm_votes, cnn_votes, facenet_votes = [], [], []

    for file in files:
        img = Image.open(BytesIO(file.file.read())).convert("RGB")

        # HOG + SVM
        gray = img.convert("L").resize((64, 64))
        feat = hog(np.array(gray), pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)) or np.all(feat == 0):
            continue
        svm_pred = int(svm.predict([feat])[0])
        svm_votes.append(svm_pred)

        # CNN
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = cnn_model(tensor)
            cnn_pred = int(out.argmax(1).item())
            cnn_votes.append(cnn_pred)

        # FaceNet
        aligned = mtcnn(img)
        if aligned is not None:
            emb = facenet(aligned.unsqueeze(0)).detach()
            sims = [torch.nn.functional.cosine_similarity(emb.squeeze(0), torch.tensor(e).squeeze(0), dim=0).item()
                    for e in facenet_data["vikas_embeddings"]]
            avg_sim = np.mean(sims)
            fn_pred = 1 if avg_sim > facenet_data["threshold"] else 0
        else:
            fn_pred = 0
        facenet_votes.append(fn_pred)

    def majority(votes):
        return max(set(votes), key=votes.count) if votes else -1

    final_vote = majority([majority(svm_votes), majority(cnn_votes), majority(facenet_votes)])

    return {
        "svm": int(majority(svm_votes)),
        "cnn": int(majority(cnn_votes)),
        "facenet": int(majority(facenet_votes)),
        "final_decision": "vikas" if final_vote == 1 else "other"
    }

@app.get("/")
def root():
    return {"message": "Face Authentication API running."}
