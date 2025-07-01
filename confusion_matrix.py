import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from models.age_model import AgeModel
from models.transformer_category_model import TransformerCategory
from data import combined_datasets  # or your actual dataset loader

def get_model_class(name):
    if name.lower() == 'agemodel':
        return AgeModel
    elif name.lower() == 'transformercategory':
        return TransformerCategory
    else:
        raise ValueError(f"Unknown model architecture: {name}")

def plot_and_save_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Confusion matrix saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Save confusion matrix for a model checkpoint.")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--out', type=str, default='confusion_matrix.png', help='Output image file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model_class = get_model_class(config['model_architecture'])
    model = model_class(config=config, pretrained=False)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load validation data
    # Assumes get_dataset returns a dict with 'val' key for val dataloader
    from train import get_dataset
    data = get_dataset(config)
    val_loader = data['val'] if isinstance(data, dict) else data.val_dataloader()

    all_preds = []
    all_targets = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[:2]  # ignore extra fields if present
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if hasattr(model, 'logits_to_pred'):
                preds = model.logits_to_pred(logits)
            elif hasattr(model, '_decode'):
                preds = model._decode(logits)
            else:
                preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds)

    class_names = [str(i) for i in range(config['num_classes'])]
    plot_and_save_confusion_matrix(cm, class_names, args.out)

if __name__ == '__main__':
    main() 