import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import numpy as np
import ast, math, os
from PIL import Image
import cv2

from data.data_utils import unnormalize
from preprocess.preprocess_utils import create_mask
from preprocess.mmpose_fill import get_keypoints_info


# Function to load an image from a given path
def load_image(path):
    img_path = path
    # img_path = os.path.join(DATA, path)
    img = Image.open(img_path)
    return img

def query_prediction_results_similarity(
        root,
        query_metadata,
        db_metadata,
        query_start,
        similarity_scores, 
        num_images=10
    ):
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    fig.tight_layout(pad=0.5)
    
    for i in range(num_images):
        idx = query_start + i
        
        # Query image
        query_img_path = query_metadata.iloc[idx]['path']
        query_img = load_image(os.path.join(root, query_img_path))
        
        # Predicted image (EXACT MATCH)
        # Get index of most similar database image
        closest_db_idx = np.argmax(similarity_scores[idx])
        predicted_img_path = db_metadata.iloc[closest_db_idx]['path']
        predicted_img = load_image(os.path.join(root, predicted_img_path))
        
        # Ground truth image (for comparison)
        ground_truth_label = query_metadata.iloc[idx]['identity']
        filtered_id_truth = db_metadata[db_metadata['identity'] == ground_truth_label]
        ground_truth_img_path = filtered_id_truth['path'].values[0]
        ground_truth_img = load_image(os.path.join(root, ground_truth_img_path))
        
        # --- Visualization Logic ---
        # Display query image
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title(f'Query: {query_metadata.iloc[idx]["identity"]}')
        axes[i, 0].axis('off')
        
        # Display predicted image (exact match)
        predicted_label = db_metadata.iloc[closest_db_idx]['identity']
        is_correct = (predicted_label == ground_truth_label)
        axes[i, 1].imshow(predicted_img)
        axes[i, 1].set_title(f'Predicted: {predicted_label}', color='green' if is_correct else 'red')
        axes[i, 1].axis('off')
        
        # Display ground truth image
        axes[i, 2].imshow(ground_truth_img)
        axes[i, 2].set_title(f'Ground Truth: {ground_truth_label}')
        axes[i, 2].axis('off')
    
    plt.show()  

def query_prediction_results(root, query_metadata, db_metadata, query_start, predictions, num_images=10):
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    fig.tight_layout(pad=0.5)
    
    for i in range(num_images):
        idx = query_start + i
        
        # Query image
        query_img_path = query_metadata.iloc[idx]['path']
        # print("query img path:", query_img_path)
        query_img = load_image(os.path.join(root, query_img_path))
        
        # Predicted image
        predicted_label = predictions[idx]
        # print("predicted label:", predicted_label)
        
        filtered_id_pred = db_metadata[db_metadata['identity'] == predicted_label] #predicted IDENTITY (NOT image necessarily)
        # print("filtered id pred:", filtered_id_pred)

        predicted_img_path = filtered_id_pred['path'].values[0]
        # print("predicted img path:", predicted_img_path)
        if 'species' in filtered_id_pred.columns:
            predicted_img_species = filtered_id_pred['species'].values[0]
        else:
            predicted_img_species = ''
        # print("predicted img species:", predicted_img_species)
        predicted_img = load_image(os.path.join(root,predicted_img_path))
        
        # Ground truth image
        ground_truth_label = query_metadata.iloc[idx]['identity'] # identity of the query image
        # print ("ground truth label:", ground_truth_label)
        filtered_id_truth = db_metadata[db_metadata['identity'] == ground_truth_label] # get that from db
        # print("filtered id truth:", filtered_id_truth)
        print

        ground_truth_img_path = filtered_id_truth['path'].values[0] #ERROR
        if 'species' in filtered_id_truth.columns:
            truth_img_species = filtered_id_truth['species'].values[0]
        else:
            truth_img_species = ''
        ground_truth_img = load_image(os.path.join(root,ground_truth_img_path))
        
        # Display query image
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title(f'Query Image: {query_metadata.iloc[idx]['identity']}')
        axes[i, 0].axis('off')

        predicted_color = 'green' if (predicted_label == ground_truth_label) else 'red'
        
        # Display predicted image
        axes[i, 1].imshow(predicted_img)
        axes[i, 1].set_title(f'Predicted: {predicted_img_species}, {predicted_label}', color=predicted_color)
        axes[i, 1].axis('off')
        
        # Display ground truth image
        axes[i, 2].imshow(ground_truth_img)
        axes[i, 2].set_title(f'Ground Truth: {truth_img_species}, {ground_truth_label}')
        axes[i, 2].axis('off')

    plt.show()


def masked_img(root, df_row=None, image_path=None, bbox=None, segmentation=None, normalized=False):
    """
    Shows masked, cropped to size of bbox
    """

    if df_row is not None:
        image_path = os.path.join(root, df_row['path'])
        bbox = df_row['bbox']
        segmentation = df_row['segmentation']
        label = df_row['identity']

    image = Image.open(image_path)

    x_min = math.floor(bbox[0])
    y_min = math.floor(bbox[1])
    w = math.ceil(bbox[2])
    h = math.ceil(bbox[3])
    bbox = [x_min, y_min, w, h]

    mask = create_mask(image.size, segmentation)

    # masked_image = np.array(cropped_image) * np.expand_dims(cropped_mask, axis=2)
    masked_image = np.array(image) * np.expand_dims(mask, axis=2)
    masked_image = Image.fromarray(masked_image.astype('uint8'))

    # Crop the image and the mask to the bounding box
    masked_image = masked_image.crop((x_min, y_min, x_min + w, y_min + h))

    # Plot the image
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # Single axis

    if normalized is not False:
        image_np = unnormalize(masked_image, normalized[0], normalized[1]).permute(1, 2, 0).numpy()  # [:3] selects RGB channels
    else:
        image_np = np.array(masked_image)   # Convert from PyTorch tensor to numpy array

    ax.imshow(image_np)
    ax.set_title(f'Label: {label}')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

def keypoints_on_img(root, df_row=None, image_path=None, bbox=None, keypoints=None):
    
    if df_row is not None:
        image_path = os.path.join(root, df_row['path'])
        bbox = df_row['bbox']
        segmentation = df_row['segmentation']
        keypoints = df_row['keypoints']
        label = df_row['identity']
        keypoint_names = get_keypoints_info()

    image = Image.open(image_path)

    # Create a matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # Single axis
    ax.imshow(image)

    # Draw the bounding box
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Draw the keypoints 
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v > 0 or v==0:  # If visibility flag is > 0, draw the keypoint
            ax.plot(x, y, 'bo')  # Blue dot for keypoints
            try:
                ax.text(x,y, keypoint_names[i//3], fontsize=6, color='white')
            except IndexError:
                print(f"Cannot access keypoint name at index {i//3}")
            

    # Show the image with annotations
    plt.axis('off')
    plt.show()

def keypoint_names_on_img(root, df_row=None, image_path=None, bbox=None, keypoints=None):
    if df_row is not None:
        image_path = os.path.join(root, df_row['path'])
        bbox = df_row['bbox']
        segmentation = df_row['segmentation']
        keypoints = df_row['keypoints']
        label = df_row['identity']

    joint_names = get_keypoints_info()

    img = cv2.imread(image_path)
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    keypoint_positions = []
    for j in range(0, len(keypoints), 3):
        kp_x, kp_y, v = keypoints[j:j+3]
        keypoint_positions.append((kp_x, kp_y))
        if v > 0:  # Only plot visible keypoints
            cv2.circle(img, (kp_x, kp_y), 3, (0, 255, 0), -1)
            cv2.putText(img, joint_names[j // 3], (kp_x + 5, kp_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)

    # # Draw skeleton lines between the keypoints
    # for link in skeleton:
    #     start_idx, end_idx = link[0] - 1, link[1] - 1  # Convert to zero-indexed
    #     if keypoints[start_idx*3+2] > 0 and keypoints[end_idx*3+2] > 0:  # Only draw if both keypoints are visible
    #         cv2.line(img, keypoint_positions[start_idx], keypoint_positions[end_idx], (0, 255, 255), 2)

    # Show the image with keypoints, skeleton, and joint names
    cv2.imwrite(f'results/output_image-{label}.jpg', img)
    #end visualization
