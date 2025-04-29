import matplotlib.pyplot as plt
import matplotlib.patches as patches

import json
import pandas as pd
import numpy as np
import ast, math, os
from PIL import Image
import cv2
import textwrap

from data.data_utils import unnormalize
from preprocess.preprocess_utils import create_mask
from preprocess.mmpose_fill import get_keypoints_info

from data.transforms import denormalize
from data.transforms import denorm_RGB_components

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
        num_images=10,
        to_save = False
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
        # Wrap text to a fixed width (20 characters)
        query_title = textwrap.fill(query_metadata.iloc[idx]["identity"], width=20)
        predicted_label = db_metadata.iloc[closest_db_idx]['identity']
        predicted_title = textwrap.fill(predicted_label, width=20)
        gt_title = textwrap.fill(ground_truth_label, width=20)

        # Display query image
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title(f'Query:\n{query_title}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Display predicted image (exact match)
        is_correct = (predicted_label == ground_truth_label)
        axes[i, 1].imshow(predicted_img)
        axes[i, 1].set_title(f'Predicted:\n{predicted_title}', fontsize=10,
                               color='green' if is_correct else 'red')
        axes[i, 1].axis('off')
        
        # Display ground truth image
        axes[i, 2].imshow(ground_truth_img)
        axes[i, 2].set_title(f'Ground Truth:\n{gt_title}', fontsize=10)
        axes[i, 2].axis('off')
    if to_save:
        return fig
    else:
        plt.show()  

def query_prediction_results_similarity_preprocessed(
        root,
        query_metadata,
        db_metadata,
        query_start,
        similarity_scores, 
        num_images=10,
        preprocess_option: int = None  # if 2, show preprocessed (masked) image in extra column
    ):
    """
    Visualizes query, predicted, and ground truth images side-by-side.
    
    - With 3 columns by default (Query, Predicted, Ground Truth).
    - If preprocess_option == 2, a fourth column shows the preprocessed (masked) image.
    - If preprocess_option >= 3, a fourth column shows keypoints drawn on the query image.
    """
    # Determine the number of columns.
    num_columns = 4 if preprocess_option in [2] or (preprocess_option is not None and preprocess_option >= 3) else 3
    
    fig, axes = plt.subplots(num_images, num_columns, figsize=(3 * num_columns, 3 * num_images))
    fig.tight_layout(pad=0.5)
    
    # If only one row, convert axes to 2D array for consistent indexing.
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i in range(num_images):
        idx = query_start + i
        query_row = query_metadata.iloc[idx]
        
        # Load query image.
        query_img = load_image(os.path.join(root, query_row['path']))
        
        # Predicted image (exact match).
        closest_db_idx = np.argmax(similarity_scores[idx])
        predicted_img_path = db_metadata.iloc[closest_db_idx]['path']
        predicted_img = load_image(os.path.join(root, predicted_img_path))
        
        # Ground truth image (first match from db_metadata with matching identity).
        ground_truth_label = query_row['identity']
        filtered_truth = db_metadata[db_metadata['identity'] == ground_truth_label]
        ground_truth_img_path = filtered_truth['path'].values[0]
        ground_truth_img = load_image(os.path.join(root, ground_truth_img_path))
        
        # Titles with wrapped text.
        query_title = textwrap.fill(query_row["identity"], width=20)
        predicted_label = db_metadata.iloc[closest_db_idx]['identity']
        predicted_title = textwrap.fill(predicted_label, width=20)
        gt_title = textwrap.fill(ground_truth_label, width=20)
        
        # --- Plotting ---
        # Query image.
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title(f'Query:\n{query_title}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Predicted image.
        is_correct = (predicted_label == ground_truth_label)
        axes[i, 1].imshow(predicted_img)
        axes[i, 1].set_title(f'Predicted:\n{predicted_title}', fontsize=10,
                               color='green' if is_correct else 'red')
        axes[i, 1].axis('off')
        
        # Ground truth image.
        axes[i, 2].imshow(ground_truth_img)
        axes[i, 2].set_title(f'Ground Truth:\n{gt_title}', fontsize=10)
        axes[i, 2].axis('off')
        
        # Extra column based on preprocess_option.
        if preprocess_option == 2:
            # Show masked (preprocessed) image.
            masked_img_arr = get_masked_image(root, query_row)
            axes[i, 3].imshow(masked_img_arr)
            axes[i, 3].set_title("Masked Image", fontsize=10)
            axes[i, 3].axis('off')
        elif preprocess_option is not None and preprocess_option >= 3:
            # Show keypoints on image.
            # Open the query image with PIL for drawing.
            image = Image.open(os.path.join(root, query_row['path']))
            ax = axes[i, 3]
            ax.imshow(image)
            
            # Draw the bounding box.
            bbox = query_row['bbox']
            if type(bbox) == str:
                bbox = json.loads(bbox)
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Draw keypoints.
            keypoints = query_row.get('keypoints', [])
            # Convert keypoints from strings to floats if necessary.
            if keypoints and isinstance(keypoints[0], str):
                keypoints = [float(k) for k in keypoints]
            keypoint_names = get_keypoints_info()
            # Expecting keypoints as a list of numbers: [x, y, v, ...]
            for j in range(0, len(keypoints), 3):
                x, y, v = keypoints[j], keypoints[j+1], keypoints[j+2]
                if v > 0:  # Visible keypoint.
                    ax.plot(x, y, 'bo')
                    try:
                        kp_name = keypoint_names[j//3]
                    except IndexError:
                        kp_name = ''
                    ax.text(x, y, kp_name, fontsize=6, color='white')
            
            ax.set_title("Keypoints", fontsize=10)
            ax.axis('off')
    
    plt.show()

def get_masked_image(root, df_row):
    """
    Process the image using the same logic as masked_img but return
    the resulting image array instead of plotting it.
    """
    # Get values from the DataFrame row
    image_path = os.path.join(root, df_row['path'])
    bbox = df_row['bbox']
    segmentation = df_row['segmentation']
    label = df_row['identity']
    
    # Open image
    image = Image.open(image_path)
    
    # Compute bounding box coordinates
    if type(bbox) == str:
        bbox = json.loads(bbox)
    x_min = math.floor(bbox[0])
    y_min = math.floor(bbox[1])
    w = math.ceil(bbox[2])
    h = math.ceil(bbox[3])
    
    # Create mask and apply it
    mask = create_mask(image.size, segmentation)
    masked_image = np.array(image) * np.expand_dims(mask, axis=2)
    masked_image = Image.fromarray(masked_image.astype('uint8'))
    
    # Crop to bounding box
    masked_image = masked_image.crop((x_min, y_min, x_min + w, y_min + h))
    return np.array(masked_image)


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
def new_query_prediction_results_similarity(
        query_raw_img,
        query_rgb_img,
        query_task_img,
        predicted_raw_img,
        predicted_rgb_img,
        predicted_task_img,
        query_identity,
        predicted_identity,
        epoch,
        preprocess_lvl,
        to_save=False
    ):
    try:
        # Convert PIL images to numpy arrays if necessary
        if isinstance(query_raw_img, Image.Image):
            query_raw_img = np.array(query_raw_img)
        if isinstance(predicted_raw_img, Image.Image):
            predicted_raw_img = np.array(predicted_raw_img)
        if isinstance(query_rgb_img, Image.Image):
            query_rgb_img = np.array(query_rgb_img)
        if isinstance(predicted_rgb_img, Image.Image):
            predicted_rgb_img = np.array(predicted_rgb_img)

        # Ensure RGB images are in correct format
        if query_raw_img.shape[2] != 3:
            query_raw_img = query_raw_img[:, :, :3]
        if predicted_raw_img.shape[2] != 3:
            predicted_raw_img = predicted_raw_img[:, :, :3]
        if query_rgb_img.shape[2] != 3:
            query_rgb_img = query_rgb_img[:, :, :3]
        if predicted_rgb_img.shape[2] != 3:
            predicted_rgb_img = predicted_rgb_img[:, :, :3]

        # Determine number of columns based on preprocess_lvl
        if preprocess_lvl == 3 and query_task_img is not None:
            num_cols = 6  # Query: raw, rgb, skeleton; Predicted: raw, rgb, skeleton
        elif preprocess_lvl == 4 and query_task_img is not None:
            num_components = query_task_img.shape[0] // 3  # Each component has 3 channels
            num_cols = 2 + 2 + (2 * num_components)  # Query: raw, rgb; Predicted: raw, rgb; + components
        elif preprocess_lvl == 5 and query_task_img is not None:
            num_heatmaps = query_task_img.shape[0]  # Each heatmap is a single channel
            num_cols = 2 + 2 + (2 * num_heatmaps)  # Query: raw, rgb; Predicted: raw, rgb; + heatmaps
        else:
            num_cols = 4  # Query: raw, rgb; Predicted: raw, rgb

        # Create figure with dynamic number of columns
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
        if num_cols == 1:
            axes = [axes]  # Ensure axes is a list for consistent indexing
        fig.tight_layout(pad=0.5)

        query_title = textwrap.fill(str(query_identity), width=20)
        predicted_title = textwrap.fill(str(predicted_identity), width=20)
        is_correct = (str(query_identity) == str(predicted_identity))

        # Plot Query Raw
        axes[0].imshow(query_raw_img)
        axes[0].set_title(f'Query Raw:\n{query_title}', fontsize=10)
        axes[0].axis('off')

        # Plot Query RGB
        axes[1].imshow(query_rgb_img)
        axes[1].set_title(f'Query Masked:\n{query_title}', fontsize=10)
        axes[1].axis('off')

        # Plot Predicted Raw
        axes[2].imshow(predicted_raw_img)
        axes[2].set_title(f'Predicted Raw:\n{predicted_title}', fontsize=10, color='green' if is_correct else 'red')
        axes[2].axis('off')

        # Plot Predicted RGB
        axes[3].imshow(predicted_rgb_img)
        axes[3].set_title(f'Predicted Masked:\n{predicted_title}', fontsize=10, color='green' if is_correct else 'red')
        axes[3].axis('off')

        # Handle task-specific channels based on preprocess_lvl
        if preprocess_lvl >= 3 and query_task_img is not None:
            if preprocess_lvl == 3:
                # Level 3: Single skeleton channel
                query_skeleton = query_task_img.cpu().numpy() if isinstance(query_task_img, torch.Tensor) else query_task_img
                predicted_skeleton = predicted_task_img.cpu().numpy() if isinstance(predicted_task_img, torch.Tensor) else predicted_task_img

                axes[4].imshow(query_skeleton, cmap='gray')
                axes[4].set_title(f'Query Skeleton:\n{query_title}', fontsize=10)
                axes[4].axis('off')

                axes[5].imshow(predicted_skeleton, cmap='gray')
                axes[5].set_title(f'Predicted Skeleton:\n{predicted_title}', fontsize=10, color='green' if is_correct else 'red')
                axes[5].axis('off')

            elif preprocess_lvl == 4:
                # Level 4: Multiple RGB components
                num_components = query_task_img.shape[0] // 3
                for i in range(num_components):
                    # Query component
                    query_component = query_task_img[3*i:3*(i+1)].cpu().numpy() if isinstance(query_task_img, torch.Tensor) else query_task_img[3*i:3*(i+1)]
                    query_component = np.transpose(query_component, (1, 2, 0))  # (C, H, W) to (H, W, C)
                    axes[4 + 2*i].imshow(query_component.astype(np.uint8))
                    axes[4 + 2*i].set_title(f'Query Component {i+1}:\n{query_title}', fontsize=10)
                    axes[4 + 2*i].axis('off')

                    # Predicted component
                    predicted_component = predicted_task_img[3*i:3*(i+1)].cpu().numpy() if isinstance(predicted_task_img, torch.Tensor) else predicted_task_img[3*i:3*(i+1)]
                    predicted_component = np.transpose(predicted_component, (1, 2, 0))  # (C, H, W) to (H, W, C)
                    axes[5 + 2*i].imshow(predicted_component.astype(np.uint8))
                    axes[5 + 2*i].set_title(f'Predicted Component {i+1}:\n{predicted_title}', fontsize=10, color='green' if is_correct else 'red')
                    axes[5 + 2*i].axis('off')

            elif preprocess_lvl == 5:
                # Level 5: Multiple heatmaps
                num_heatmaps = query_task_img.shape[0]
                for i in range(num_heatmaps):
                    # Query heatmap
                    query_heatmap = query_task_img[i].cpu().numpy() if isinstance(query_task_img, torch.Tensor) else query_task_img[i]
                    if query_heatmap.max() > 0:
                        query_heatmap = query_heatmap / query_heatmap.max()  # Normalize to [0, 1]
                    axes[4 + 2*i].imshow(query_heatmap, cmap='hot')
                    axes[4 + 2*i].set_title(f'Query Heatmap {i+1}:\n{query_title}', fontsize=10)
                    axes[4 + 2*i].axis('off')

                    # Predicted heatmap
                    predicted_heatmap = predicted_task_img[i].cpu().numpy() if isinstance(predicted_task_img, torch.Tensor) else predicted_task_img[i]
                    if predicted_heatmap.max() > 0:
                        predicted_heatmap = predicted_heatmap / predicted_heatmap.max()  # Normalize to [0, 1]
                    axes[5 + 2*i].imshow(predicted_heatmap, cmap='hot')
                    axes[5 + 2*i].set_title(f'Predicted Heatmap {i+1}:\n{predicted_title}', fontsize=10, color='green' if is_correct else 'red')
                    axes[5 + 2*i].axis('off')

        plt.suptitle(f'Epoch {epoch}', fontsize=12)
        if to_save:
            return fig
        else:
            plt.show()
            return None
    except Exception as e:
        print(f"Error in query_prediction_results_similarity: {e}")
        return None
    

def viz_skeleton(mean, std, dataloader):

    batch = next(iter(dataloader))
    images, labels = batch['img'], batch['label']

    print("Batch Size:", images.size(0))
    print("Image Shape:", images.shape)
    print("Labels Shape:", labels.shape)

    # Plot the images and their skeletons side by side
    fig, axes = plt.subplots(4, 2, figsize=(10, 20))  # 4 rows, 2 columns (RGB and skeleton for each image)
    axes = axes.flatten()

    for i in range(4):  # Assuming batch size is at least 4
        # Unnormalize the RGB part
        rgb_image = unnormalize(images[i], mean, std).permute(1, 2, 0).numpy()  # First 3 channels (RGB)
        skeleton_image = images[i][3].cpu().numpy()  # 4th channel (Skeleton)

        # Plot RGB image
        ax_rgb = axes[i * 2]
        ax_rgb.imshow(np.clip(rgb_image, 0, 1))  # Clip values to [0, 1] for valid image display
        ax_rgb.set_title(f'RGB Image - Label: {labels[i].item()}')
        ax_rgb.axis('off')

        # Plot Skeleton image (grayscale)
        ax_skel = axes[i * 2 + 1]
        ax_skel.imshow(skeleton_image, cmap='gray')  # Plot skeleton channel as a grayscale image
        ax_skel.set_title(f'Skeleton Channel - Label: {labels[i].item()}')
        ax_skel.axis('off')

    plt.tight_layout()
    plt.show()

def viz_heatmaps(mean, std, dataloader):
    kp_names = get_keypoints_info()

    batch = next(iter(dataloader))
    images, labels = batch['img'], batch['label']

    print("Batch Size:", images.size(0))
    print("Image Shape:", images.shape)  # Should be [Batch, 3 + num_heatmaps, H, W]
    print("Labels Shape:", labels.shape)

    # Define the number of heatmaps (channels) after the RGB channels
    num_heatmaps = images.shape[1] - 3  # Subtract RGB channels

    # Plot the images and their heatmaps side by side
    fig, axes = plt.subplots(4, num_heatmaps + 1, figsize=(20, 5))  # 4 rows, num_heatmaps + 1 columns
    axes = axes.flatten()

    for i in range(4):  # Assuming batch size is at least 4
        # Extract the RGB channels
        image_np = images[i][:3].numpy()  # Take first 3 channels (RGB)
        image_np = denormalize(image_np, mean, std)  # Denormalize the RGB channels
        image_np = np.transpose(image_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for plotting
        label = labels[i].item()  # Convert label tensor to Python scalar using .item()

        # Plot RGB image
        ax_rgb = axes[i * (num_heatmaps + 1)]
        ax_rgb.imshow(image_np)  # Remove clipping here, as image_np should be in [0, 1] range now
        ax_rgb.set_title(f'{labels[i].item()}')
        ax_rgb.axis('off')

        # Plot each heatmap (grayscale)
        for j in range(num_heatmaps):
            heatmap = images[i][3 + j].cpu().numpy()  # Get the heatmap (after the 3 RGB channels)
            
            # Normalize the heatmap if necessary (if values are too small/large)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()  # Normalize to [0, 1]

            # Plot the heatmap
            ax_heatmap = axes[i * (num_heatmaps + 1) + (j + 1)]
            ax_heatmap.imshow(heatmap, cmap='hot')  # Plot heatmap as a grayscale image
            ax_heatmap.set_title(kp_names[j], fontsize=7) 
            ax_heatmap.axis('off')

    plt.tight_layout()
    plt.show()

def viz_components(mean, std, dataloader):
    # Fetch a batch of images (assuming batch size is 8 for example)
    batch = next(iter(dataloader))
    images, labels = batch['img'], batch['label']
    print("Batch Size:", images.size(0))
    print("Image Shape:", images.shape)  # Expected shape: (batch_size, channels, height, width)
    print("Labels Shape:", labels.shape)

    for i in range(images.size(0)):  # Loop through each image in the batch
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))  # Create 2x3 grid for each image
        axes = axes.flatten()

        # Extract the full image with RGB + component channels
        full_image = images[i].numpy()  # Shape: (channels, height, width)

        # Denormalize the full image (including RGB and component channels)
        full_image_denorm = denorm_RGB_components(full_image, mean, std)

        # Plot RGB image (first 3 channels)
        rgb_image = full_image_denorm[:3]  # Extract the first 3 channels (RGB)
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Convert to (H, W, C) for plotting
        
        ax = axes[0]
        ax.imshow(rgb_image.astype(np.uint8))  # Convert back to integer type for proper visualization
        ax.set_title(f'Label: {labels[i].item()} - RGB')
        ax.axis('off')

        # Plot component channels (next 5 sets of 3 channels each)
        for j in range(5):  # Assuming 5 components, each with 3 channels
            # Extract the next set of 3 channels for the current component
            component_channels = full_image_denorm[3 + (j * 3): 3 + (j * 3) + 3]  # Extract the component channels
            component_image = np.transpose(component_channels, (1, 2, 0))  # Convert to (H, W, C) for plotting

            ax = axes[j + 1]  # Move to the next subplot (j + 1 because axes[0] is used for RGB)
            ax.imshow(component_image.astype(np.uint8))  # Convert back to integer type for proper visualization
            ax.set_title(f'Component {j + 1}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

def viz_imgs(mean, std, dataloader, denorm=True):
    # Fetch a batch of images (assuming batch size is 8 for example)
    batch = next(iter(dataloader))
    images,labels = batch['img'], batch['label']
    print("Batch Size:", images.size(0))
    print("Image Shape:", images.shape)
    print("Labels Shape:", labels.shape)

    # Extract images and labels from the batch
    images, labels = batch  # This assumes batch is structured as (images, labels)

    # Plot the images
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()

    for i in range(8):  # Assuming batch size is 8
        # image_np = unnormalize(images[i], mean, std).permute(1, 2, 0).numpy()  # [:3] selects RGB channels

        image_np = images[i].numpy()  # Convert from PyTorch tensor to numpy array
        if denorm:
            image_np = denormalize(image_np, mean, std)  # Denormalize the image

        image_np = np.transpose(image_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for plotting
        label = labels[i].item()  # Convert label tensor to Python scalar using .item()
        ax = axes[i]
        ax.imshow(image_np)
        ax.set_title(f'Label: {label}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()