import torch
from torchvision import transforms, models
from PIL import Image
import sys

classes = ['O', 'R']  # Organic, Recyclable

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model/saved_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_path = sys.argv[1]
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    print(f"Prediction: {classes[predicted.item()]}")
