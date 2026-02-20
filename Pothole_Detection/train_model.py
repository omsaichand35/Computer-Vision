import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transformer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#Load Dataset
train_data = datasets.ImageFolder("../pothole_dataset/train", transform=transform)
val_data = datasets.ImageFolder("../pothole_dataset/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

#Load pretrained model
model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "pothole_dataset.pth")
print("Model saved")