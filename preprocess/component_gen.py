# idea from https://www.sciencedirect.com/science/article/pii/S1470160X21011717#b0200
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from preprocess.preprocess_utils import create_mask
from PIL import Image


def component_generation_module(image, bbox, keypoints, keypoint_labels, mask = True, segmentation = None):
    """
    Generate component-based cropped regions from keypoints based on the 23 Wildlife keypoint skeleton structure.
    
    Parameters:
        image (np.array): The input image.
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        keypoints (list of tuples): List of (x, y) keypoints coordinates.
        keypoint_labels (list of str): List of keypoint labels corresponding to the keypoints.
        
    Returns:
        cropped_images (dict): Dictionary with component names as keys and cropped, masked images as values.
    """
    # Map keypoint labels to their coordinates
    label_to_coords = {}
    num_keypoints = len(keypoints) // 3  # Every keypoint has x, y, v

    for idx in range(num_keypoints):
        x = keypoints[3 * idx]
        y = keypoints[3 * idx + 1]
        v = keypoints[3 * idx + 2]  # Visibility flag
        label = keypoint_labels[idx]
        # if v > 0 and x > 0 and y > 0:
        label_to_coords[label] = (int(x), int(y))
        # else:
        #     print(f"Keypoint {label} not visible or invalid")
        #     label_to_coords[label] = None  # Keypoint not visible or invalid

    # print(label_to_coords)
    
    # Define components and their associated keypoints
    components = {
        "HEAD": [
            "Head_Mid_Top",
            "Eye_Left",
            "Eye_Right",
            "Mouth_Front_Top",
            "Mouth_Back_Left",
            "Mouth_Back_Right",
            "Mouth_Front_Bottom",
        ],
        "WING_LEFT": [
            "Shoulder_Left",
            "Elbow_Left",
            "Wrist_Left",
        ],
        "WING_RIGHT": [
            "Shoulder_Right",
            "Elbow_Right",
            "Wrist_Right",
        ],
        "BODY_LEGS": [
            "Torso_Mid_Back",
            "Hip_Left",
            "Hip_Right",
            "Knee_Left",
            "Knee_Right",
            "Ankle_Left",
            "Ankle_Right",
        ],
        "TAIL": [
            "Tail_Top_Back",
            "Tail_Mid_Back",
            "Tail_End_Back",
        ],
    }
    
    cropped_images = {}
    # image_height, image_width = image.shape[:2]
    xmin_bbox, ymin_bbox, width_bbox, height_bbox = bbox
    xmax_bbox = xmin_bbox + width_bbox
    ymax_bbox = ymin_bbox + height_bbox

    # We want just the masked image (no background)
    mask = create_mask((image.shape[1], image.shape[0]), segmentation)
    masked_image = np.array(image) * np.expand_dims(mask, axis=2)

    padding = int(width_bbox * 0.05) if width_bbox<height_bbox else int(height_bbox * 0.025)

    # For each category:
    for category, keypoint_names in components.items():
        # Collect keypoints for this component
        component_coords = [
            label_to_coords[label]
            for label in keypoint_names
            if label in label_to_coords and label_to_coords[label] is not None
        ]
        # print(f"Component {category} keypoints: {component_coords}")
        
        if component_coords:
             # Calculate bounding box around these keypoints
            xs = [x for x, y in component_coords]
            ys = [y for x, y in component_coords]

            if len(component_coords) == 1:
                padding *= 2
            #     # Single keypoint -- expand the bounding box
            #     xmin_comp = int(max(xmin_bbox, xs-padding))
            #     xmax_comp = int(min(xmax_bbox, xs+padding))
            #     ymin_comp = int(max(ymin_bbox, ys-padding))
            #     ymax_comp = int(min(ymax_bbox, ys+padding))
            # else:
            xmin_comp = int(max(xmin_bbox, min(xs)-padding))
            xmax_comp = int(min(xmax_bbox, max(xs)+padding))
            ymin_comp = int(max(ymin_bbox, min(ys)-padding))
            ymax_comp = int(min(ymax_bbox, max(ys)+padding))

            center_x = (xmin_comp + xmax_comp) // 2
            center_y = (ymin_comp + ymax_comp) // 2

            half_size = max(xmax_comp - xmin_comp, ymax_comp - ymin_comp) // 2
            
            xmin_comp, ymin_comp, xmax_comp, ymax_comp = get_square_crop_box(center_x, center_y, half_size, xmin_bbox, ymin_bbox, xmax_bbox, ymax_bbox)

            # Crop the image - using masked_image instead of image
            cropped_img = masked_image[int(ymin_comp):int(ymax_comp), int(xmin_comp):int(xmax_comp)]
            cropped_images[category] = cropped_img

        else:
            # No keypoints found for this component
            print(f"No keypoints found for {category}")
            cropped_images[category] = None  # Empty result for missing keypoints
    
    return cropped_images


def get_square_crop_box(center_x, center_y, half_size, xmin_bbox, ymin_bbox, xmax_bbox, ymax_bbox ):
    """
    Get a square crop box centered around a given point, constrained within the bounding box and image boundaries.

    Returns:
        tuple: (xmin, ymin, xmax, ymax) coordinates of the square crop.
    """

    # Ensure the crop stays within the bounding box
    xmin = max(xmin_bbox, center_x - half_size)
    xmax = min(xmax_bbox, center_x + half_size)
    ymin = max(ymin_bbox, center_y - half_size)
    ymax = min(ymax_bbox, center_y + half_size)


    # Crop inside bbox -- size:
    width = xmax - xmin
    height = ymax - ymin
    if width > height:
        diff = width - height
        # push the center lower
        ymin = ymin - diff // 2
        ymax = ymax + diff - diff // 2
    elif width < height:
        diff = height - width
        xmin = xmin - diff // 2
        xmax = xmax + diff - diff // 2

    return (xmin, ymin, xmax, ymax)