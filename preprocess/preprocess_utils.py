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

# def create_skeleton_channel(keypoints, connections, height, width, sigma=2, thickness=2):
#     """
#     Create a 4th channel for the model input representing the skeleton.
    
#     Args:
#         keypoints (list): List of flattened COCO-style keypoints (x, y, visibility).
#         connections (list): List of (start_idx, end_idx) for limbs, based on keypoint indices.
#         height (int): Height of the image.
#         width (int): Width of the image.
#         sigma (int): Gaussian blur for keypoints.
#         thickness (int): Thickness of the drawn limbs.
    
#     Returns:
#         skeleton_channel (np.array): The skeleton channel.
#     """

#     heatmap = np.zeros((height, width), dtype=np.float32)
#     skeleton_channel = np.zeros((height, width), dtype=np.float32)
    
#     # Create heatmaps for keypoints
#     for i in range(0, len(keypoints), 3):
#         x, y, visibility = float(keypoints[i]), float(keypoints[i+1]), int(keypoints[i+2])
        
#         # Skip keypoints that are not visible or invalid
#         if visibility == 0 or x < 0 or y < 0:
#             continue

#         # Create a Gaussian blob centered at (x, y)
#         for h in range(height):
#             for w in range(width):
#                 heatmap[h, w] += np.exp(-((w - x) ** 2 + (h - y) ** 2) / (2 * sigma ** 2))

#     # Draw limbs on the skeleton channel
#     for (start_idx, end_idx) in connections:
#         start_x, start_y, start_vis = keypoints[(start_idx - 1) * 3:(start_idx - 1) * 3 + 3]
#         end_x, end_y, end_vis = keypoints[(end_idx - 1) * 3:(end_idx - 1) * 3 + 3]

#         # Draw line only if both keypoints are visible
#         if start_vis > 0 and end_vis > 0:
#             start_point = (int(start_x), int(start_y))
#             end_point = (int(end_x), int(end_y))
#             cv2.line(skeleton_channel, start_point, end_point, 1, thickness)

#     # Combine keypoints heatmap and skeleton lines
#     skeleton_channel += heatmap
#     skeleton_channel = np.clip(skeleton_channel, 0, 1)  # Normalize to [0, 1] range

#     return skeleton_channel

def create_skeleton_channel(keypoints, connections, height, width, sigma=2, thickness=2, crop_to_bbox=None):
    """
    Create a skeleton channel efficiently using vectorized operations, with optional cropping.
    
    Args:
        keypoints (list): List of flattened COCO-style keypoints (x, y, visibility).
        connections (list): List of (start_idx, end_idx) for limbs.
        height (int): Height of the image or cropped region.
        width (int): Width of the image or cropped region.
        sigma (int): Gaussian blur for keypoints.
        thickness (int): Thickness of the drawn limbs.
        crop_to_bbox (tuple, optional): (x, y, w, h) to crop to; if None, uses full image.
    
    Returns:
        skeleton_channel (np.array): The skeleton channel.
    """
    # Determine dimensions based on cropping
    if crop_to_bbox is not None:
        x, y, w, h = [int(round(coord)) for coord in crop_to_bbox]
        heatmap = np.zeros((h, w), dtype=np.float32)
        skeleton_channel = np.zeros((h, w), dtype=np.float32)
    else:
        x, y = 0, 0  # No offset for full image
        h, w = height, width
        heatmap = np.zeros((height, width), dtype=np.float32)
        skeleton_channel = np.zeros((height, width), dtype=np.float32)

    # Adjust keypoints if cropping
    if crop_to_bbox is not None:
        adjusted_keypoints = []
        for i in range(0, len(keypoints), 3):
            kp_x, kp_y, vis = keypoints[i:i+3]
            adjusted_x, adjusted_y = kp_x - x, kp_y - y
            if 0 <= adjusted_x < w and 0 <= adjusted_y < h:
                adjusted_keypoints.extend([adjusted_x, adjusted_y, vis])
            else:
                adjusted_keypoints.extend([0, 0, 0])  # Invisible outside bbox
    else:
        adjusted_keypoints = keypoints  # Use original keypoints for full image

    # Vectorized heatmap generation
    visible_keypoints = [(float(adjusted_keypoints[i]), float(adjusted_keypoints[i+1])) 
                         for i in range(0, len(adjusted_keypoints), 3) 
                         if int(adjusted_keypoints[i+2]) > 0 and adjusted_keypoints[i] >= 0 and adjusted_keypoints[i+1] >= 0]
    if visible_keypoints:
        x_coords, y_coords = zip(*visible_keypoints)
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        for kp_x, kp_y in visible_keypoints:
            heatmap += np.exp(-((x_grid - kp_x) ** 2 + (y_grid - kp_y) ** 2) / (2 * sigma ** 2))

    # Draw limbs
    for start_idx, end_idx in connections:
        start_x, start_y, start_vis = adjusted_keypoints[(start_idx - 1) * 3:(start_idx - 1) * 3 + 3]
        end_x, end_y, end_vis = adjusted_keypoints[(end_idx - 1) * 3:(end_idx - 1) * 3 + 3]
        if start_vis > 0 and end_vis > 0:
            start_point = (int(start_x), int(start_y))
            end_point = (int(end_x), int(end_y))
            cv2.line(skeleton_channel, start_point, end_point, 1, thickness)

    skeleton_channel += heatmap
    skeleton_channel = np.clip(skeleton_channel, 0, 1)
    return skeleton_channel

def create_multichannel_heatmaps(keypoints, height, width, bbox_width, bbox_height, sigma=25):
    """
    Create individual heatmaps for each keypoint using Gaussian blobs.
    
    Args:
        keypoints (list): List of flattened COCO-style keypoints (x, y, visibility).
        height (int): Height of the output heatmap.
        width (int): Width of the output heatmap.
        bbox_width (float): Bounding box width.
        bbox_height (float): Bounding box height.
        sigma (float): Base sigma factor relative to bbox size.
    
    Returns:
        List[np.ndarray]: One heatmap per keypoint.
    """
    heatmaps = []
    sigma = sigma * max(bbox_width, bbox_height) / 1000.0
    tmp_size = int(3 * sigma)

    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        if v == 0 or x < 0 or y < 0:
            heatmaps.append(heatmap)
            continue

        x_int = int(x)
        y_int = int(y)

        # Bounding box for the gaussian
        ul = [max(0, x_int - tmp_size), max(0, y_int - tmp_size)]
        br = [min(width, x_int + tmp_size + 1), min(height, y_int + tmp_size + 1)]

        size = 2 * tmp_size + 1
        g = np.fromfunction(
            lambda y_, x_: np.exp(-((x_ - tmp_size) ** 2 + (y_ - tmp_size) ** 2) / (2 * sigma ** 2)),
            (size, size),
            dtype=np.float32
        )

        g_x_min = max(0, tmp_size - x_int)
        g_y_min = max(0, tmp_size - y_int)
        g_x_max = g_x_min + br[0] - ul[0]
        g_y_max = g_y_min + br[1] - ul[1]

        heatmap[ul[1]:br[1], ul[0]:br[0]] = g[g_y_min:g_y_max, g_x_min:g_x_max]
        heatmaps.append(heatmap)

    return heatmaps

# def create_multichannel_heatmaps(keypoints, height, width, bbox_width, bbox_height, sigma=25):
#     """
#     Create individual heatmaps for each keypoint in the given image dimensions.
    
#     Args:
#         keypoints (list): List of flattened COCO-style keypoints (x, y, visibility).
#         height (int): Height of the image.
#         width (int): Width of the image.
#         sigma (int): Gaussian spread for the keypoints.

#     Returns:
#         keypoint_heatmaps (list): A list of heatmaps, one for each keypoint.
#     """
#     keypoint_heatmaps = []
#     if bbox_width > bbox_height:
#         sigma = bbox_width * 0.05
#     else:
#         sigma = bbox_height * 0.05

#     # Create heatmaps for keypoints
#     for i in range(0, len(keypoints), 3):
#         x, y, visibility = float(keypoints[i]), float(keypoints[i+1]), int(keypoints[i+2])
        
#         # Initialize an empty heatmap for the current keypoint
#         heatmap = np.zeros((height, width), dtype=np.float32)

#         # Skip keypoints that are not visible or invalid
#         if visibility == 0 or x < 0 or y < 0:
#             keypoint_heatmaps.append(heatmap)
#             continue

#         # Create a Gaussian blob centered at (x, y) by computung the value for every pixel
#         for h in range(height):
#             for w in range(width):
#                 heatmap[h, w] = np.exp(-((w - x) ** 2 + (h - y) ** 2) / (2 * sigma ** 2))

#         # Normalize the heatmap to range [0, 1]
#         heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
#         keypoint_heatmaps.append(heatmap)

#     return keypoint_heatmaps