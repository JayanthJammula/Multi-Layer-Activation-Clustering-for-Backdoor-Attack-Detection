from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from src.utils import pattern_craft, add_backdoor

# Argument parser (optional for extensibility, defaults are set)
parser = argparse.ArgumentParser(description='PyTorch Backdoor Attack Crafting')
parser.add_argument('--num_attacks', type=int, default=500, help='Number of attacks to craft')
parser.add_argument('--source_class', type=int, default=1, help='Source class for the backdoor')
parser.add_argument('--target_class', type=int, default=7, help='Target class for the backdoor')
parser.add_argument('--pattern_type', type=str, default='static', help='Type of backdoor pattern (e.g., chess_board, cross, 4pixel)')
parser.add_argument('--perturbation_size', type=float, default=3/255, help='Size of the perturbation')
args = parser.parse_args()

# Parameters
NUM_OF_ATTACKS = args.num_attacks
SC = args.source_class
TC = args.target_class
PATTERN_TYPE = args.pattern_type
PERT_SIZE = args.perturbation_size

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed for reproducibility (optional)
random.seed()

# Load datasets
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Craft perturbation
pert = pattern_craft((3, 32, 32), PATTERN_TYPE, PERT_SIZE)

# Crafting training backdoor images
print('==> Crafting training backdoor images...')
ind_train = [i for i, label in enumerate(list(trainset.targets)) if label == SC]

if NUM_OF_ATTACKS > len(ind_train):
    raise ValueError("NUM_OF_ATTACKS exceeds the number of available samples for the source class.")

ind_train = np.random.choice(ind_train, NUM_OF_ATTACKS, replace=False)

train_images_attacks = []
train_labels_attacks = []
for i in ind_train:
    train_images_attacks.append(add_backdoor(trainset.__getitem__(i)[0], pert).unsqueeze(0))
    train_labels_attacks.append(torch.tensor([TC], dtype=torch.long))
train_images_attacks = torch.cat(train_images_attacks, dim=0)
train_labels_attacks = torch.cat(train_labels_attacks, dim=0)

# Crafting test backdoor images
print('==> Crafting test backdoor images...')
ind_test = [i for i, label in enumerate(list(testset.targets)) if label == SC]

test_images_attacks = []
test_labels_attacks = []
for i in ind_test:
    test_images_attacks.append(add_backdoor(testset.__getitem__(i)[0], pert).unsqueeze(0))
    test_labels_attacks.append(torch.tensor([TC], dtype=torch.long))
test_images_attacks = torch.cat(test_images_attacks, dim=0)
test_labels_attacks = torch.cat(test_labels_attacks, dim=0)

# Ensure 'attacks' directory exists
print('==> Saving backdoor images...')
os.makedirs('attacks', exist_ok=True)

# Save crafted data
train_attacks = {'image': train_images_attacks, 'label': train_labels_attacks}
test_attacks = {'image': test_images_attacks, 'label': test_labels_attacks}

torch.save(train_attacks, './attacks/train_attacks')
torch.save(test_attacks, './attacks/test_attacks')
torch.save(ind_train, './attacks/ind_train')
torch.save(PATTERN_TYPE, './attacks/pattern_type')
torch.save(pert, './attacks/pert')

print('Backdoor attack crafting complete. Data saved in ./attacks/')
