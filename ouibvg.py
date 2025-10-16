from __future__ import absolute_import, print_function

import torch
from src.resnet import ResNet18  # Ensure this matches your ResNet implementation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # Load model
    print("Loading model...")
    model = ResNet18()
    model.load_state_dict(torch.load('./contam/model_contam.pth', map_location=device))
    model = model.to(device)
    print("Model loaded successfully.")

    # Print all named modules
    print("\n[INFO] Printing all layers in the model:")
    for name, module in model.named_modules():
        print(name)

if __name__ == "__main__":
    main()

