"""
Goes through a directory of images and their annotations in dataframe format and runs pose estimation on each cropped image. 
Saves each image with its keypoints and skeleton drawn on it.
Finally, returns the updated dataframe (with the skeleton information).
"""

import os
from mmpose.apis import MMPoseInferencer
import cv2


def fill_keypoints(df, image_dir=""):

    # Initialize the MMPoseInferencer 
    inferencer = MMPoseInferencer(
        pose2d='td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256',
        pose2d_weights='https://download.openmmlab.com/mmpose/v1/animal_2d_keypoint/topdown_heatmap/animal_kingdom/td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256-566feff5_20230519.pth',
        device='cpu'
    )

    # Loop through each row and update with keypoints
    for row in df.iterrows():
        image_id = row['image_id']
        bbox = row['bbox']  # [x1, y1, width, height]
        # segmentation = row['segmentation']
        img_path = os.path.join(image_dir, row['path'])

        if not os.path.exists(img_path):
            print(f"File does not exist: {img_path}")
            continue

        img = cv2.imread(img_path)

        # Crop the image using the bounding box
        x, y, w, h = [int(v) for v in bbox]

        cropped_img = img[y:y+h, x:x+w]
        
    # # Save the cropped image temporarily
    # cropped_img_path = 'temp_cropped_img.jpg'
    # cv2.imwrite(cropped_img_path, cropped_img)
    
    result_generator = inferencer(cropped_img)
    # end crop

    # Perform pose estimation using the bounding box from the annotation
    result = next(result_generator)
    
    if result['predictions']:
        # Assuming a single instance detected
        instance_data = result['predictions'][0][0]
        keypoints = instance_data['keypoints']
        keypoint_scores = instance_data['keypoint_scores']

        # Convert keypoints and scores to the required format
        formatted_keypoints = []
        for i, (keypoint, score) in enumerate(zip(keypoints, keypoint_scores)):
            # handle cropping
            original_x = int(keypoint[0]) + x
            original_y = int(keypoint[1]) + y
            if score > 0.25:  # threshold to consider the keypoint
                formatted_keypoints.extend([original_x, original_y, 2])
            else:
                formatted_keypoints.extend([original_x, original_y, 0])

        # Update the annotation with keypoints
        row['keypoints'] = formatted_keypoints
        row['num_keypoints'] = len(keypoint_scores) # keep all keypoints, even if not visible

        print(f"Processed image: {image_id}")

    return df
