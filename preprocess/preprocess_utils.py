import cv2
import numpy as np

def create_mask(image_size, segmentation):
        """
        Creates a binary mask based on the segmentation of the object.

        Args:
            image_size (tuple): Size of the original image.
            segmentation (list): COCO-style segmentation of the object.

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
