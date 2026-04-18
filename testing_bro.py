import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ✅ Model definition (same as training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*56*56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load trained model
model = SimpleCNN(num_classes=4).to(device)
model.load_state_dict(torch.load("alz_model.pth", map_location=device))
model.eval()

# ✅ Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ✅ Path to your MRI scan (change this!)
img_path = r"C:\Users\Sai Vaibhav Vulli\Jadal\alzhimers\no_2.jpg"

# Load and preprocess image
image = Image.open(img_path)
input_tensor = transform(image).unsqueeze(0).to(device)

# ✅ Predict
with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)

class_names = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
print("Predicted class:", class_names[pred.item()])
