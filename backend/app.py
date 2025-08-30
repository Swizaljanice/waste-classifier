from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
import torch
import io

# Define classes (same as training)
CLASSES = ['Organic', 'Recyclable']

# Initialize FastAPI
app = FastAPI()

# Allow frontend React (localhost:3000) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("model/saved_model.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing pipeline (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    input_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted.item()]

    return {"prediction": predicted_class}
