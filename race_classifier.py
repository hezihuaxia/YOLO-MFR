import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 人种类别
class_names = ['Asian', 'Black', 'Indian', 'Other', 'White']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("race_classifier_full.pth", map_location=device)  # 若用GPU改成 cuda
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Imagenet 的均值和标准差
                         [0.229, 0.224, 0.225])
])

def predict_race(image_path):

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return class_names[pred_idx], confidence
