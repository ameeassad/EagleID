"""
Goes through a directory of images and their annotations in COCO format and runs pose estimation on each cropped image. 
Saves each image with its keypoints and skeleton drawn on it.
Finally, saves the updated annotations back to a file (with the skeleton information).
"""

import json, os
from mmpose.apis import MMPoseInferencer

import matplotlib.pyplot as plt
import cv2
import numpy as np
import ast
from PIL import Image

# Load your COCO annotations file
with open('/Users/amee/Documents/code/master-thesis/AgeClassifier/annot/modified_train_annotations.json', 'r') as f:
    coco_data = json.load(f)

skeleton =[]
joint_names = []

for cat in coco_data['categories']:
    if not cat.get('keypoints'):
        cat['keypoints'] = [
            "Head_Mid_Top",
            "Eye_Left",
            "Eye_Right",
            "Mouth_Front_Top",
            "Mouth_Back_Left",
            "Mouth_Back_Right",
            "Mouth_Front_Bottom",
            "Shoulder_Left",
            "Shoulder_Right",
            "Elbow_Left",
            "Elbow_Right",
            "Wrist_Left",
            "Wrist_Right",
            "Torso_Mid_Back",
            "Hip_Left",
            "Hip_Right",
            "Knee_Left",
            "Knee_Right",
            "Ankle_Left",
            "Ankle_Right",
            "Tail_Top_Back",
            "Tail_Mid_Back",
            "Tail_End_Back"
        ]
        cat['skeleton'] = [
            [2,1],
            [3,1],
            [4,5],
            [4,6],
            [7,5],
            [7,6],
            [1,14],
            [14,21],
            [21,22],
            [22,23],
            [1,8],
            [1,9],
            [8,10],
            [9,11],
            [10,12],
            [11,13],
            [21,15],
            [21,16],
            [15,17],
            [16,18],
            [17,19],
            [18,20]
        ]
        skeleton = cat['skeleton']
        joint_names = cat['keypoints']

        
# Initialize the MMPoseInferencer 
inferencer = MMPoseInferencer(
    pose2d='td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256',
    pose2d_weights='https://download.openmmlab.com/mmpose/v1/animal_2d_keypoint/topdown_heatmap/animal_kingdom/td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256-566feff5_20230519.pth',
    device='cpu'
)

# Loop through each annotation and update with keypoints
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    bbox = annotation['bbox']  # [x1, y1, width, height]
    segmentation = annotation['segmentation']
    
    # Assuming the corresponding image is loaded or accessible via its path
    image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
    img_path = f"/Users/amee/Library/CloudStorage/GoogleDrive-ameeassad@gmail.com/My Drive/artportalen_goeag/{image_info['file_name']}.jpg"

    if not os.path.exists(img_path):
        print(f"File does not exist: {img_path}")
        continue

    img = cv2.imread(img_path)

    # Crop the image using the bounding box
    x, y, w, h = [int(v) for v in bbox]
    cropped_img = img[y:y+h, x:x+w]
        
    # Save the cropped image temporarily
    cropped_img_path = 'temp_cropped_img.jpg'
    cv2.imwrite(cropped_img_path, cropped_img)
    result_generator = inferencer(cropped_img_path)
    # end crop

    # Perform pose estimation using the bounding box from the annotation
    # result_generator = inferencer(img_path, bboxes=[bbox])
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
            if score > 0.5:  # threshold to consider the keypoint
                formatted_keypoints.extend([original_x, original_y, 2])
            else:
                formatted_keypoints.extend([original_x, original_y, 0])
            #end crop

            # if score > 0.5:  # threshold to consider the keypoint
            #     formatted_keypoints.extend([int(keypoint[0]), int(keypoint[1]), 2])
            # else:
            #     formatted_keypoints.extend([int(keypoint[0]), int(keypoint[1]), 0])

        # Update the annotation with keypoints
        annotation['keypoints'] = formatted_keypoints
        annotation['num_keypoints'] = len(keypoint_scores) # keep all keypoints, even if not visible

        # Save the updated annotations to a file after each iteration
        with open('results/intermediate_annotations.json', 'w') as f:
            json.dump(coco_data, f, indent=4)

        print(f"Processed image: {image_id}")

        # # Delete the below for visualisation
        # img = cv2.imread(img_path)
        # x, y, w, h = [int(v) for v in bbox]
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # keypoint_positions = []
        # for j in range(0, len(formatted_keypoints), 3):
        #     kp_x, kp_y, v = formatted_keypoints[j:j+3]
        #     keypoint_positions.append((kp_x, kp_y))
        #     if v > 0:  # Only plot visible keypoints
        #         cv2.circle(img, (kp_x, kp_y), 3, (0, 255, 0), -1)
        #         cv2.putText(img, joint_names[j // 3], (kp_x + 5, kp_y - 5), 
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)

        # # Draw skeleton lines between the keypoints
        # for link in skeleton:
        #     start_idx, end_idx = link[0] - 1, link[1] - 1  # Convert to zero-indexed
        #     if formatted_keypoints[start_idx*3+2] > 0 and formatted_keypoints[end_idx*3+2] > 0:  # Only draw if both keypoints are visible
        #         cv2.line(img, keypoint_positions[start_idx], keypoint_positions[end_idx], (0, 255, 255), 2)

        # # Show the image with keypoints, skeleton, and joint names
        # cv2.imwrite(f'results/output_image-{image_id}.jpg', img)
        # #end visualization

# Save the updated annotations back to a file
with open('results/final_annotations.json', 'w') as f:
    json.dump(coco_data, f, indent=4)
