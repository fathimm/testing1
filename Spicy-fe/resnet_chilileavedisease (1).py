import torch
import torchvision
import time
import numpy as np
from numpy import transpose
from matplotlib import pyplot as plt
from warnings import filterwarnings as fw
from sklearn.model_selection import train_test_split
import os
import shutil
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from PIL import Image

fw("ignore")

"""# Menampilkan Gambar dari Tensor (imshow)"""

def imshow(img : torch.Tensor, *args, **kwargs) -> None:
    img = img / 2 + 0.5  # unormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(transpose(npimg, (1, 2, 0)))
    plt.show()

"""# Seed"""

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Inisialisasi seed untuk CUDA (jika digunakan)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

"""# Hyper Parameter"""

EPOCH = 110
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

"""# Data Transformation"""

fixed_size = (256, 256)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(fixed_size),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.GaussianBlur(kernel_size=3),
    torchvision.transforms.Normalize(*stats, inplace=True)
])

"""# Dataset Splitting"""

# Path ke direktori dataset utama di Raspberry Pi
dataset_path = 'C:/Users/USER/.jupyter/lab/workspaces/testing1/dataset_modifikasi' # Ganti dengan path dataset lokal di Raspberry Pi

# Path untuk direktori output train, val, dan test
output_path = 'C:/Users/USER/.jupyter/lab/workspaces/testing1/DatasetModifikasiNew_Split'  # Ganti dengan path output yang diinginkan
os.makedirs(output_path, exist_ok=True)

# Mendapatkan daftar semua kelas (folder) di dataset
classes = os.listdir(dataset_path)

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    # Membagi dataset untuk setiap kelas menjadi train, val, dan test
    train_path, test_val_path = train_test_split(os.listdir(class_path), test_size=0.2, random_state=0)
    val_path, test_path = train_test_split(test_val_path, test_size=0.5, random_state=0)

    # Membuat direktori output untuk setiap set (train, val, test)
    train_output = os.path.join(output_path, 'train', class_name)
    val_output = os.path.join(output_path, 'val', class_name)
    test_output = os.path.join(output_path, 'test', class_name)

    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    # Memindahkan file ke direktori output
    for file_name in train_path:
        shutil.copy(os.path.join(class_path, file_name), os.path.join(train_output, file_name))

    for file_name in val_path:
        shutil.copy(os.path.join(class_path, file_name), os.path.join(val_output, file_name))

    for file_name in test_path:
        shutil.copy(os.path.join(class_path, file_name), os.path.join(test_output, file_name))

"""# DataLoader Setup"""

# Training and validation sets
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(output_path, 'train'), transform=data_transform)
total_length = len(train_dataset)

# Tentukan rasio untuk pembagian
train_ratio = 0.8  # contoh: 80% untuk pelatihan
valid_ratio = 0.2  # contoh: 20% untuk validasi

# Hitung panjang untuk pelatihan dan validasi
train_length = int(train_ratio * total_length)
valid_length = total_length - train_length

# Gunakan panjang yang dihitung untuk pembagian
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_length, valid_length])

# Test set
test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(output_path, 'test'), transform=data_transform)

# DATALOAD ---> assign raw datasets into mini-batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Total TRAIN dataset: {}.\nFor mini-batch settings, there will be {} batches, with {} data for each batch".
      format(len(train_dataset), len(train_loader), BATCH_SIZE))

print("Total VALIDATION dataset: {}.\nFor mini-batch settings, there will be {} batches, with {} data for each batch".
      format(len(valid_dataset), len(valid_loader), BATCH_SIZE))

print("Total TEST dataset: {}.\nFor mini-batch settings, there will be {} batches, with {} data for each batch".
      format(len(test_dataset), len(test_loader), BATCH_SIZE))

"""# Dataset Label"""

cabe_label = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Powdery Mildew', 'White Fly', 'Yellowish']

"""# Displaying Training Set Samples"""

dataiter = iter(train_loader)
images, labels = next(dataiter)
valid_labels = [cabe_label[i] for i in labels if i < len(cabe_label)]

imshow(img=torchvision.utils.make_grid(images))
print("LABELS (every 8 labels, going down):")
for i in range(0, len(valid_labels), 8):
    print(", ".join(valid_labels[i:i+8]))

"""# ResNet Architecture Implementation"""

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, n_class=6):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_class)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

model = ResNet()
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Save the model checkpoint
PATH = 'C:/Users/USER/.jupyter/lab/workspaces/testing1/resnet.pth'  # Ganti dengan path yang diinginkan untuk menyimpan model
torch.save(model.state_dict(), PATH)

# """# Training Loop"""
#
# for epoch in range(EPOCH):
#     model.train()
#     total_loss = 0
#     for i, (images, labels) in enumerate(train_loader):
#         if torch.cuda.is_available():
#             images, labels = images.cuda(), labels.cuda()
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#             print(f"Epoch [{epoch + 1}/{EPOCH}], Loss: {running_loss / len(train_loader):.4f}")
#
#     model.eval()
#     total_correct = 0
#     with torch.no_grad():
#         for images, labels in valid_loader:
#             if torch.cuda.is_available():
#                 images, labels = images.cuda(), labels.cuda()
#
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total_correct += (predicted == labels).sum().item()
#
      # accuracy = total_correct / len(valid_dataset)
#      print(f"Validation Loss: {val_loss/len(valid_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")
#
#       for epoch in range(EPOCH):
#            train(model, train_loader, criterion, optimizer, epoch)
#             validate(model, valid_loader, criterion)
#
#     if epoch % 10 == 0:
#         torch.save(model.state_dict(), PATH)


"""# Training & Validation"""

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCH}], Loss: {running_loss/len(train_loader):.4f}")

def validate(model, valid_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Loss: {val_loss/len(valid_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

for epoch in range(EPOCH):
    train(model, train_loader, criterion, optimizer, epoch)
    validate(model, valid_loader, criterion)

"""# Testing Loop"""

model.eval()
total_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / len(test_dataset)
print(f'Test Accuracy: {accuracy:.2f}%')

"""# Confusion Matrix and Classification Report"""

y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("Classification Report:")
print(classification_report(y_true, y_pred))

"""# Visualizing Confusion Matrix"""

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
