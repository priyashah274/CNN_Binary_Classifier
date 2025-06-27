# utils.py
import torch
from torchvision import transforms
from PIL import Image
from CNNModel import CNNClassifier

def load_model(model_path="model/binary_classifier.pth", device="cpu"):
    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(image, model, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image)
        pred_probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1).item()

    classes = ["Cat", "Dog"]
    return classes[pred_class], pred_probs[0][pred_class].item()
