from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
from PIL import Image
import torch
import io

app = FastAPI()

# CORS to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["Organic", "Recyclable"]

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("model/saved_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = CLASSES[predicted.item()]

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
