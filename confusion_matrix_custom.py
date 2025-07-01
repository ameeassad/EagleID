import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from models.age_model import AgeModel
from data.artportalen_goleag import ArtportalenDataModule
import yaml

config_path = '/Users/amee/Documents/code/master-thesis/EagleID/configs/config-hpc-artportalen.yml'
ckpt_path = '/Users/amee/Documents/code/master-thesis/EagleID/checkpoints/agemodel2.ckpt'
val_img_dir = '/Users/amee/Documents/code/master-thesis/datasets/artportalen_goeag'
val_csv = '/Users/amee/Documents/code/master-thesis/AgeClassifier/annot/final_train_sep_sightings.csv'
out_path = 'confusion_matrix-yA.png'

def plot_and_save_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Confusion matrix saved to {out_path}")

if __name__ == '__main__':
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override file paths for local run
    config['dataset'] = val_img_dir
    config['cache_path'] = val_csv
    # Use val_csv for both train and val for this script
    train_csv = val_csv
    
    # Get settings from config
    preprocess_lvl = config.get('preprocess_lvl', 2)
    batch_size = config.get('batch_size', 32)
    img_size = config.get('img_size', 224)
    mean = config['transforms']['mean'][0] if isinstance(config['transforms']['mean'], list) else config['transforms']['mean']
    std = config['transforms']['std'][0] if isinstance(config['transforms']['std'], list) else config['transforms']['std']
    num_classes = config.get('num_classes', 5)
    class_names = [str(i) for i in range(num_classes)]

    # Load model
    model = AgeModel(config=config, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load Artportalen validation dataset using the DataModule
    data_module = ArtportalenDataModule(
        data_dir=val_img_dir,
        preprocess_lvl=preprocess_lvl,
        batch_size=batch_size,
        size=img_size,
        mean=mean,
        std=std,
        test=True
    )
    data_module.setup_from_csv(train_csv=train_csv, val_csv=val_csv)
    val_dataset = data_module.val_dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[:2]
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if hasattr(model, '_decode'):
                preds = model._decode(logits)
            else:
                preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds)
    plot_and_save_confusion_matrix(cm, class_names, out_path) 