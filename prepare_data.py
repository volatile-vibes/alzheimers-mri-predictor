import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Path to your dataset folder
DATA_DIR = "C:\Users\Sai Vaibhav Vulli\Downloads\ds007446-main\ds007446-main"


X = []
y = []

# Loop through subjects
for sub in os.listdir(DATA_DIR):
    anat_path = os.path.join(DATA_DIR, sub, "anat")
    if not os.path.exists(anat_path):
        continue
    for file in os.listdir(anat_path):
        if file.endswith("_T1w.nii.gz"):
            img = nib.load(os.path.join(anat_path, file))
            data = img.get_fdata()

            # Convert to tensor and resize to 96x96x48
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(tensor, size=(96,96,48), mode="trilinear", align_corners=False)
            X.append(resized.numpy())

            # Dummy labels for now (alternate 0/1)
            y.append(len(y) % 2)

X = np.vstack(X)   # shape: (samples, 1, 96, 96, 48)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])
