# idea from https://www.sciencedirect.com/science/article/pii/S1470160X21011717#b0200

import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

def component_generation_module(image, bbox, keypoints, keypoint_labels, num_clusters=3):
    """
    Generate component-based cropped regions from keypoints.
    
    Parameters:
        image (np.array): The input image.
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        keypoints (list of tuples): List of (x, y) keypoints coordinates.
        keypoint_labels (list of str): List of keypoint labels corresponding to the keypoints.
        num_clusters (int): Number of clusters to group keypoints (default is 3).
        
    Returns:
        cropped_images (list of np.array): List of cropped component regions.
        cluster_centers (list of tuples): List of cluster centers used for cropping.
    """
    
    # Extract keypoint coordinates
    keypoint_coords = np.array([(x, y) for (x, y) in keypoints])
    
    # Step 1: Perform K-means clustering on the keypoints
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(keypoint_coords)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Step 2: Calculate the bird's length (bounding box diagonal) to define crop size
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    target_length = np.sqrt(bbox_width ** 2 + bbox_height ** 2)
    
    # Step 3: Define the cropping size based on target length
    crop_size = int(target_length * 0.3)  # Adjustable scale factor for cropping regions
    
    cropped_images = []
    
    # Step 4: Generate cropping regions for each cluster
    for center in cluster_centers:
        center_x, center_y = int(center[0]), int(center[1])
        
        # Define the cropping region around the cluster center
        xmin = max(0, center_x - crop_size // 2)
        xmax = min(image.shape[1], center_x + crop_size // 2)
        ymin = max(0, center_y - crop_size // 2)
        ymax = min(image.shape[0], center_y + crop_size // 2)
        
        # Crop the region from the original image
        cropped_img = image[ymin:ymax, xmin:xmax]
        
        # Add the cropped region to the list
        cropped_images.append(cropped_img)
        
        # Optional: Display the cropped region (for debugging)
        # plt.imshow(cropped_img)
        # plt.show()
    
    return cropped_images, cluster_centers

# Example usage of the CGM
def example_cgm_usage():
    # Load example image
    image = cv.imread("example_image.jpg")
    
    # Define bounding box (xmin, ymin, xmax, ymax)
    bbox = (50, 50, 300, 400)
    
    # Segmentation mask (binary mask, not used in this simplified example)
    segmentation = np.zeros_like(image[:, :, 0])
    
    # Example keypoints and labels (x, y) positions of detected keypoints
    keypoints = [(120, 100), (150, 80), (200, 300), (250, 220), (180, 150)]
    keypoint_labels = ["right_eye", "left_eye", "left_ankle", "right_ankle", "beak"]
    
    # Call CGM function
    cropped_images, cluster_centers = component_generation_module(image, bbox, segmentation, keypoints, keypoint_labels, num_clusters=3)
    
    # Display results
    for i, cropped_img in enumerate(cropped_images):
        plt.imshow(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))
        plt.title(f"Cropped Region {i+1}")
        plt.show()
    
    print("Cluster Centers:", cluster_centers)


# Function to crop square regions around specified keypoints
def crop_square_regions(image, bbox, keypoints, keypoint_labels, crop_size):
    """
    Crop square regions around specified keypoints, allowing overlap, and ensuring full coverage.

    Parameters:
        image (np.array): The input image.
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        keypoints (dict): Dictionary of keypoint coordinates with labels as keys and (x, y) as values.
        keypoint_labels (list of str): List of keypoint labels to consider.
        crop_size (int): The size of the square cropping box around each keypoint.
        
    Returns:
        cropped_images (dict): Dictionary with keypoint labels as keys and cropped images as values.
    """
    
    xmin, ymin, xmax, ymax = bbox
    image_height, image_width = image.shape[:2]

    cropped_images = {}
    remaining_regions = [(xmin, ymin, xmax, ymax)]

    for label in keypoint_labels:
        if label in keypoints and keypoints[label] is not None:
            keypoint_x, keypoint_y = keypoints[label]
            
            # Calculate the square cropping region around the keypoint
            center_x = int(keypoint_x)
            center_y = int(keypoint_y)
            
            # Define the region ensuring it doesn't go out of image bounds
            xmin_region = max(0, center_x - crop_size // 2)
            xmax_region = min(image_width, center_x + crop_size // 2)
            ymin_region = max(0, center_y - crop_size // 2)
            ymax_region = min(image_height, center_y + crop_size // 2)
            
            # Crop the square region from the image
            cropped_img = image[ymin_region:ymax_region, xmin_region:xmax_region]
            cropped_images[label] = cropped_img

            # Subtract this region from the available remaining region
            remaining_regions = update_remaining_regions(
                remaining_regions, xmin_region, ymin_region, xmax_region, ymax_region
            )
        
        else:
            # If keypoint is missing, assign a fallback region (from remaining area)
            if remaining_regions:
                # Pick the first available remaining region
                r_xmin, r_ymin, r_xmax, r_ymax = remaining_regions.pop(0)
                
                # Adjust the region to fit the square size
                fallback_xmax = min(r_xmin + crop_size, r_xmax)
                fallback_ymax = min(r_ymin + crop_size, r_ymax)
                
                # Crop the fallback square region from the image
                cropped_img = image[r_ymin:fallback_ymax, r_xmin:fallback_xmax]
                cropped_images[label] = cropped_img

    return cropped_images

def update_remaining_regions(remaining_regions, xmin, ymin, xmax, ymax):
    """
    Update the list of remaining regions after cropping a region.

    Parameters:
        remaining_regions (list of tuples): List of remaining regions (xmin, ymin, xmax, ymax).
        xmin, ymin, xmax, ymax (int): Coordinates of the cropped region.

    Returns:
        list: Updated list of remaining regions.
    """
    new_remaining_regions = []

    for region in remaining_regions:
        r_xmin, r_ymin, r_xmax, r_ymax = region

        # Split the remaining region based on the cropped area
        if xmin > r_xmin:
            new_remaining_regions.append((r_xmin, r_ymin, xmin, r_ymax))
        if xmax < r_xmax:
            new_remaining_regions.append((xmax, r_ymin, r_xmax, r_ymax))
        if ymin > r_ymin:
            new_remaining_regions.append((r_xmin, r_ymin, r_xmax, ymin))
        if ymax < r_ymax:
            new_remaining_regions.append((r_xmin, ymax, r_xmax, r_ymax))

    return new_remaining_regions

# Example usage
def example_crop_usage():
    # Load example image
    image = cv.imread("example_image.jpg")
    
    # Define bounding box (xmin, ymin, xmax, ymax)
    bbox = (50, 50, 300, 400)
    
    # Example keypoints and labels (x, y) positions of detected keypoints
    keypoints = {
        "tail": (120, 350),
        "left_wrist": (200, 200),
        "right_wrist": None,  # No keypoint for right_wrist
        "head": (90, 100)
    }
    
    keypoint_labels = ["tail", "left_wrist", "right_wrist", "head"]
    
    # Define the size of the square region to crop around each keypoint (e.g., 100x100 pixels)
    crop_size = 100
    
    # Call the cropping function
    cropped_images = crop_square_regions(image, bbox, keypoints, keypoint_labels, crop_size)
    
    # Display the results
    for label, cropped_img in cropped_images.items():
        plt.imshow(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))
        plt.title(f"Cropped Region: {label}")
        plt.show()


if __name__ == "__main__":

    # Example usage of the cropping function
    example_crop_usage()