from __future__ import absolute_import, print_function

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.resnet import ResNet18


def safe_torch_load(path: Path, map_location=None):
    """Load torch artifacts with a weights_only flag when available."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract backdoor activations for Spectral Signature analysis.')
    parser.add_argument('--attacks-path', default='attacks/train_attacks', help='Path to the saved train_attacks tensor bundle.')
    parser.add_argument('--checkpoint', default='contam/model_contam.pth', help='Path to the contaminated model checkpoint.')
    parser.add_argument('--output-dir', default='features_backdoor', help='Directory to store extracted activation batches.')
    parser.add_argument('--layers', nargs='+', default=['layer1.0.conv1', 'layer2.0.conv2', 'layer4.1.conv2', 'linear'], help='Model layers to hook for feature dumping.')
    parser.add_argument('--batch-size', type=int, default=32, help='Dataloader batch size.')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of dataloader workers.')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pinned memory for the dataloader.')
    return parser.parse_args()


class Hook:
    """Registers a forward hook and streams batch activations to disk."""

    def __init__(self, module, layer_name: str, output_dir: Path, suffix: str = 'backdoor'):
        self.layer_name = layer_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self.batch_idx = 0
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):  # pylint: disable=unused-argument
        batch_features = output.detach().cpu().numpy()
        batch_path = self.output_dir / f'{self.layer_name}_{self.suffix}_batch_{self.batch_idx}.npy'
        print(f'[HOOK] {self.layer_name}: saving {batch_features.shape} -> {batch_path.name}')
        np.save(batch_path, batch_features)
        self.batch_idx += 1

    def close(self):
        self.hook.remove()


def extract_features(loader: DataLoader, model: torch.nn.Module, device: torch.device):
    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(loader, total=len(loader), desc='Forward passes', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            model(inputs)


def attach_hooks(model: torch.nn.Module, layer_names, output_dir: Path):
    hooks = {}
    available_layers = {name: module for name, module in model.named_modules()}
    missing_layers = []

    for layer_name in layer_names:
        module = available_layers.get(layer_name)
        if module is None:
            missing_layers.append(layer_name)
            continue
        hooks[layer_name] = Hook(module, layer_name, output_dir)

    if missing_layers:
        print(f'[WARN] Missing layers: {missing_layers}')
    if not hooks:
        raise RuntimeError('No valid hooks registered; check --layers argument and model definition.')
    return hooks


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    attacks_path = Path(args.attacks_path)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not attacks_path.exists():
        raise FileNotFoundError(f'Attack bundle not found: {attacks_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    train_attacks = safe_torch_load(attacks_path, map_location=device)
    if not {'image', 'label'}.issubset(train_attacks.keys()):
        raise KeyError('Expected keys "image" and "label" in the attack bundle.')
    print(f'Backdoor data shape: {train_attacks["image"].shape}, labels: {torch.unique(train_attacks["label"]).tolist()}')

    attack_dataset = TensorDataset(train_attacks['image'], train_attacks['label'])
    attack_loader = DataLoader(
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

    hooks = attach_hooks(model, args.layers, output_dir)

    print('Extracting features for backdoor training images...')
    extract_features(attack_loader, model, device)

    for hook in hooks.values():
        hook.close()

    print('Feature extraction for backdoor attacks completed!')


if __name__ == '__main__':
    main()
