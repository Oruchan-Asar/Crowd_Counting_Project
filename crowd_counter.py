import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
from scipy.io import loadmat
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Define the custom dataset class for crowd counting
class CrowdCountingDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images_path = os.path.join(data_path, 'images')
        self.ground_truth_path = os.path.join(data_path, 'ground-truth')

        self.image_filenames = sorted(os.listdir(self.images_path))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.images_path, image_filename)
        ground_truth_filename = 'GT_' + os.path.splitext(image_filename)[0] + '.mat'
        ground_truth_path = os.path.join(self.ground_truth_path, ground_truth_filename)

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        ground_truth = self.load_ground_truth(ground_truth_path)

        return image, ground_truth

    def load_ground_truth(self, ground_truth_path):
        mat = loadmat(ground_truth_path)
        ground_truth = mat['image_info'][0][0]['number'][0][0]
        ground_truth = np.float32(ground_truth)  # Convert to a supported type
        return torch.tensor(ground_truth, dtype=torch.float32)

# Define the CNN model for crowd counting
class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 288 * 512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Define the testing loop
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the paths to your dataset
train_data_path = './train_data'
test_data_path = './test_data'

# Define the image size for resizing
image_size = (576, 1024)

# Create the dataset loaders
train_dataset = CrowdCountingDataset(
    train_data_path,
    transform=transforms.Compose([
        Resize(image_size),
        ToTensor()
    ])
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CrowdCountingDataset(
    test_data_path,
    transform=transforms.Compose([
        Resize(image_size),
        ToTensor()
    ])
)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model and optimizer
model = CrowdCounter().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model
train_loss_values = []
test_loss_values = []

# Training and testing loops
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    test_loss = test(model, test_dataloader, criterion, device)

    avg_train_loss = train_loss / len(train_dataloader)
    train_loss_values.append(avg_train_loss)

    avg_test_loss = train_loss / len(test_dataloader)
    test_loss_values.append(avg_test_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'crowd_counter_model.pth')

# Plot train and test loss values
epochs = range(1, num_epochs + 1)

plt.plot(epochs, train_loss_values, 'b', label='Train Loss')
plt.plot(epochs, test_loss_values, 'r', label='Test Loss')
plt.title('Train and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
