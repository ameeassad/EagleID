import cv2
import numpy as np
import os
import pickle
import json
import pandas as pd
import ast


def create_mask(image_size, segmentation):
        """
        Creates a *binary mask* based on the segmentation of the object.

        Args:
            image_size (tuple): Size of the original image.
            segmentation (list): COCO-style segmentation of the object (flattened x y values).

        Returns:
            np.array: Binary mask of the object.
        """
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        for seg in segmentation:
            poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        return mask

def create_skeleton_channel(keypoints, connections, height, width, sigma=2, thickness=2):
    """
    Create a 4th channel for the model input representing the skeleton.
    
    Args:
        keypoints (list): List of flattened COCO-style keypoints (x, y, visibility).
        connections (list): List of (start_idx, end_idx) for limbs, based on keypoint indices.
        height (int): Height of the image.
        width (int): Width of the image.
        sigma (int): Gaussian blur for keypoints.
        thickness (int): Thickness of the drawn limbs.
    
    Returns:
        skeleton_channel (np.array): The skeleton channel.
    """
    # Initialize heatmap and skeleton channel
    if heatmap:
        keypoint_heatmaps = []
    else:
        heatmap = np.zeros((height, width), dtype=np.float32)
        skeleton_channel = np.zeros((height, width), dtype=np.float32)
    
    # Create heatmaps for keypoints
    for i in range(0, len(keypoints), 3):
        x, y, visibility = float(keypoints[i]), float(keypoints[i+1]), int(keypoints[i+2])
        
        # Skip keypoints that are not visible or invalid
        if visibility == 0 or x < 0 or y < 0:
            continue

        # Create a Gaussian blob centered at (x, y)
        for h in range(height):
            for w in range(width):
                heatmap[h, w] += np.exp(-((w - x) ** 2 + (h - y) ** 2) / (2 * sigma ** 2))

    # Draw limbs on the skeleton channel
    for (start_idx, end_idx) in connections:
        start_x, start_y, start_vis = keypoints[(start_idx - 1) * 3:(start_idx - 1) * 3 + 3]
        end_x, end_y, end_vis = keypoints[(end_idx - 1) * 3:(end_idx - 1) * 3 + 3]

        # Draw line only if both keypoints are visible
        if start_vis > 0 and end_vis > 0:
            start_point = (int(start_x), int(start_y))
            end_point = (int(end_x), int(end_y))
            cv2.line(skeleton_channel, start_point, end_point, 1, thickness)

    # Combine keypoints heatmap and skeleton lines
    skeleton_channel += heatmap
    skeleton_channel = np.clip(skeleton_channel, 0, 1)  # Normalize to [0, 1] range

    return skeleton_channel

def create_multichannel_heatmaps(keypoints, height, width, sigma=25):
    """
    Create individual heatmaps for each keypoint in the given image dimensions.
    
    Args:
        keypoints (list): List of flattened COCO-style keypoints (x, y, visibility).
        height (int): Height of the image.
        width (int): Width of the image.
        sigma (int): Gaussian spread for the keypoints.

    Returns:
        keypoint_heatmaps (list): A list of heatmaps, one for each keypoint.
    """
    keypoint_heatmaps = []

    # Create heatmaps for keypoints
    for i in range(0, len(keypoints), 3):
        x, y, visibility = float(keypoints[i]), float(keypoints[i+1]), int(keypoints[i+2])
        
        # Initialize an empty heatmap for the current keypoint
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Skip keypoints that are not visible or invalid
        if visibility == 0 or x < 0 or y < 0:
            keypoint_heatmaps.append(heatmap)
            continue

        # Create a Gaussian blob centered at (x, y)
        for h in range(height):
            for w in range(width):
                heatmap[h, w] = np.exp(-((w - x) ** 2 + (h - y) ** 2) / (2 * sigma ** 2))

        # Normalize the heatmap to range [0, 1]
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        keypoint_heatmaps.append(heatmap)

    return keypoint_heatmaps

def create_heatmaps():
    """
    ["Head_Mid_Top","Eye_Left","Eye_Right","Mouth_Front_Top","Mouth_Back_Left",
            "Mouth_Back_Right","Mouth_Front_Bottom","Shoulder_Left","Shoulder_Right",
            "Elbow_Left","Elbow_Right","Wrist_Left","Wrist_Right","Torso_Mid_Back",
            "Hip_Left","Hip_Right","Knee_Left","Knee_Right","Ankle_Left","Ankle_Right",
            "Tail_Top_Back","Tail_Mid_Back","Tail_End_Back"]
    """




# def cache_load_seg(cache_path, row):
#     with open(cache_path, 'rb') as cache_file:
#         cached_data = pickle.load(cache_file)
#         updated_row = row.to_dict()
#         updated_row.update(cached_data)
#     return updated_row

# def cache_load_keypoint(cache_path, row):
#     with open(cache_path, 'rb') as cache_file:
#         cached_data = pickle.load(cache_file)
#         if 'keypoints' in cached_data:
#             updated_row = row.to_dict()
#             updated_row.update(cached_data)
#     return updated_row


# def cache_save(updated_row, cache_path):
#     # Save the segmentation data to cache
#     cached_data = {
#         'segmentation': updated_row['segmentation'],
#         'height': updated_row['height'],
#         'width': updated_row['width'],
#         'bbox': updated_row['bbox'],
#         'area': updated_row['area'],
#         'iscrowd': updated_row['iscrowd'],
#     }
#     if os.path.exists(cache_path):
#         # Update existing cache
#         with open(cache_path, 'rb') as cache_file:
#             existing_data = pickle.load(cache_file)
#         existing_data.update(cached_data)
#         with open(cache_path, 'wb') as cache_file:
#             pickle.dump(existing_data, cache_file)
#     else:
#         # Ensure the directory exists
#         cache_dir = os.path.dirname(cache_path)
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
#         # Create new cache
#         with open(cache_path, 'wb') as cache_file:
#             pickle.dump(cached_data, cache_file)


