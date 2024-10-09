import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# Function to load an image from a given path
def load_image(path):
    img_path = path
    # img_path = os.path.join(DATA, path)
    img = Image.open(img_path)
    return img

def display_results(query_metadata, db_metadata, query_start, predictions, num_images=10):
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    fig.tight_layout(pad=0.5)
    
    for i in range(num_images):
        idx = query_start + i
        
        # Query image
        query_img_path = query_metadata.iloc[idx]['path']
        # print("query img path:", query_img_path)
        query_img = load_image(query_img_path)
        
        # Predicted image
        predicted_label = predictions[i]
        # print("predicted label:", predicted_label)
        filtered_id_pred = db_metadata[db_metadata['identity'] == predicted_label]
        # print("filtered id pred:", filtered_id_pred)

        predicted_img_path = filtered_id_pred['path'].values[0]
        # print("predicted img path:", predicted_img_path)
        predicted_img_species = filtered_id_pred['species'].values[0]
        # print("predicted img species:", predicted_img_species)
        predicted_img = load_image(predicted_img_path)
        
        # Ground truth image
        ground_truth_label = query_metadata.iloc[idx]['identity'] # identity of the query image
        # print ("ground truth label:", ground_truth_label)
        filtered_id_truth = db_metadata[db_metadata['identity'] == ground_truth_label] # get that from db
        # print("filtered id truth:", filtered_id_truth)
        print

        ground_truth_img_path = filtered_id_truth['path'].values[0] #ERROR
        truth_img_species = filtered_id_truth['species'].values[0]
        ground_truth_img = load_image(ground_truth_img_path)
        
        # Display query image
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title(f'Query Image: {query_metadata.iloc[idx]['identity']}')
        axes[i, 0].axis('off')
        
        # Display predicted image
        axes[i, 1].imshow(predicted_img)
        axes[i, 1].set_title(f'Predicted: {predicted_img_species}, {predicted_label}')
        axes[i, 1].axis('off')
        
        # Display ground truth image
        axes[i, 2].imshow(ground_truth_img)
        axes[i, 2].set_title(f'Ground Truth: {truth_img_species}, {ground_truth_label}')
        axes[i, 2].axis('off')

    plt.show()