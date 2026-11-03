import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Reproducibility
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "MNIST"
MODEL_DIR = BASE_DIR / "src" / "python" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device("cpu")
print("Training device:", device)

# MNIST input: 28x28 grayscale images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("\nDownloading/loading MNIST dataset...")

train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


# Neural network: 784 -> 32 -> 10
class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MNISTNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def evaluate():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy, total_loss / len(test_loader)


print("\n===== MNIST TRAINING =====")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    accuracy, test_loss = evaluate()

    print(
        f"\nEpoch {epoch + 1}: "
        f"Train Loss = {running_loss / len(train_loader):.4f}, "
        f"Test Loss = {test_loss:.4f}, "
        f"Test Accuracy = {accuracy:.2f}%\n"
    )


# Save trained model
model_path = MODEL_DIR / "mnist_fc_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": 784,
    "hidden_size": 32,
    "output_size": 10,
    "epochs": EPOCHS
}, model_path)

final_accuracy, _ = evaluate()

print("===== TRAINING COMPLETE =====")
print(f"Final test accuracy: {final_accuracy:.2f}%")
print("Model saved to:", model_path)
