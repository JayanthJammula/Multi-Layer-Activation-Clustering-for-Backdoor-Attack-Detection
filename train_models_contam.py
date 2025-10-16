from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import sys
import numpy as np
import time
from src.resnet import ResNet18


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")
net = ResNet18().to(device)  # Ensure model is on GPU

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (with backdoor)')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Preparation
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Load in attack data
if not os.path.isdir('attacks'):
    print('Attack images not found, please craft attack images first!')
    sys.exit(0)

train_attacks = torch.load('./attacks/train_attacks', weights_only=False)
test_attacks = torch.load('./attacks/test_attacks', weights_only=False)
ind_train = torch.load('./attacks/ind_train', weights_only=False)

# Poison the training set
image_dtype = trainset.data.dtype
train_images_attacks = np.rint(np.transpose(train_attacks['image'].numpy() * 255, [0, 2, 3, 1])).astype(image_dtype)
trainset.data = np.concatenate((trainset.data, train_images_attacks))
trainset.targets = np.concatenate((trainset.targets, train_attacks['label']))
trainset.data = np.delete(trainset.data, ind_train, axis=0)
trainset.targets = np.delete(trainset.targets, ind_train, axis=0)

# Load the datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
attackloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_attacks['image'], test_attacks['label']), batch_size=100, shuffle=False, num_workers=4)

# Model and Optimizer
print('==> Building model..')
net = ResNet18()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training Loop
def train(epoch):
    print(f"\n[INFO] Training Epoch: {epoch}")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(trainloader)}: Loss {train_loss / (batch_idx + 1):.4f}, Accuracy: {100. * correct / total:.2f}%")
    epoch_time = time.time() - start_time
    acc = 100. * correct / total
    print(f"[INFO] Train Accuracy: {acc:.3f}% | Epoch Time: {epoch_time:.2f}s")
    return net

def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"[INFO] Test Accuracy: {acc:.3f}%")

def test_attack(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in attackloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"[INFO] Attack Success Rate: {acc:.3f}%")

# Main Training Loop
if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 10):
        print(f"\n[INFO] Starting Epoch {epoch}")
        model_contam = train(epoch)
        test(epoch)
        test_attack(epoch)

        # Save model checkpoint
        if not os.path.isdir('contam'):
            os.mkdir('contam')
        torch.save(model_contam.state_dict(), './contam/model_contam.pth')
        print(f"[INFO] Model checkpoint saved for epoch {epoch}")
