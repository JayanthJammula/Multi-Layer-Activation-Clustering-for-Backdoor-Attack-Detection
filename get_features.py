"""
Feature Extraction for Selected Layers
"""

from __future__ import absolute_import, print_function

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.resnet import ResNet18


def safe_torch_load(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract clean and backdoor activations for SS/AC defenses.')
    parser.add_argument('--data-root', default='data', help='Root directory for CIFAR-10 data.')
    parser.add_argument('--attacks-dir', default='attacks', help='Directory containing crafted attack tensors.')
    parser.add_argument('--checkpoint', default='contam/model_contam.pth', help='Path to the contaminated model checkpoint.')
    parser.add_argument('--clean-output-dir', default='features', help='Directory to store clean activation batches.')
    parser.add_argument('--backdoor-output-dir', default='features_backdoor', help='Directory to store backdoor activation batches.')
    parser.add_argument('--layers', nargs='+', default=['layer1.0.conv1', 'layer2.0.conv2', 'layer4.1.conv2', 'linear'], help='Model layers to hook for feature dumping.')
    parser.add_argument('--batch-size', type=int, default=32, help='Dataloader batch size.')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of dataloader workers.')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pinned memory for the dataloaders.')
    return parser.parse_args()


class Hook:
    def __init__(self, module, layer_name: str, output_dir: Path, suffix: str):
        self.layer_name = layer_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self.batch_idx = 0
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):  # pylint: disable=unused-argument
        batch_features = output.detach().cpu().numpy()
        batch_path = self.output_dir / f'{self.layer_name}_{self.suffix}_batch_{self.batch_idx}.npy'
        print(f'[HOOK] {self.layer_name} ({self.suffix}): {batch_features.shape} -> {batch_path.name}')
        np.save(batch_path, batch_features)
        self.batch_idx += 1

    def close(self):
        self.hook.remove()


def attach_hooks(model: torch.nn.Module, layer_names, output_dir: Path, suffix: str):
    hooks = {}
    available_layers = {name: module for name, module in model.named_modules()}
    missing_layers = []

    for layer_name in layer_names:
        module = available_layers.get(layer_name)
        if module is None:
            missing_layers.append(layer_name)
            continue
        hooks[layer_name] = Hook(module, layer_name, output_dir, suffix)

    if missing_layers:
        print(f'[WARN] Missing layers: {missing_layers}')
    if not hooks:
        raise RuntimeError('No valid hooks registered; check --layers argument and model definition.')
    return hooks


def extract_features(loader: DataLoader, model: torch.nn.Module, device: torch.device):
    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(loader, total=len(loader), desc='Forward passes', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            model(inputs)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_root = Path(args.data_root)
    attacks_dir = Path(args.attacks_dir)
    checkpoint_path = Path(args.checkpoint)
    clean_output_dir = Path(args.clean_output_dir)
    backdoor_output_dir = Path(args.backdoor_output_dir)

    required_files = {
        'train_attacks': attacks_dir / 'train_attacks',
        'ind_train': attacks_dir / 'ind_train',
    }
    for name, file_path in required_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f'Missing required file {name}: {file_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

    train_attacks = safe_torch_load(required_files['train_attacks'], map_location=device)
    if not {'image', 'label'}.issubset(train_attacks.keys()):
        raise KeyError('Expected keys "image" and "label" in train_attacks bundle.')
    attack_dataset = TensorDataset(train_attacks['image'], train_attacks['label'])

    ind_train = safe_torch_load(required_files['ind_train'])
    ind_train = np.asarray(ind_train)
    train_data = np.array(trainset.data)
    train_targets = np.array(trainset.targets)
    train_data = np.delete(train_data, ind_train, axis=0)
    train_targets = np.delete(train_targets, ind_train, axis=0)
    trainset.data = train_data
    trainset.targets = train_targets.tolist()

    clean_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    backdoor_loader = DataLoader(
        attack_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    print('Loading model...')
    model = ResNet18()
    state_dict = safe_torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print('Model loaded successfully.')

    print('Extracting features for clean training images...')
    clean_hooks = attach_hooks(model, args.layers, clean_output_dir, suffix='clean')
    extract_features(clean_loader, model, device)
    for hook in clean_hooks.values():
        hook.close()

    print('Extracting features for backdoor training images...')
    backdoor_hooks = attach_hooks(model, args.layers, backdoor_output_dir, suffix='backdoor')
    extract_features(backdoor_loader, model, device)
    for hook in backdoor_hooks.values():
        hook.close()

    print('Feature extraction and saving completed!')


if __name__ == '__main__':
    main()
