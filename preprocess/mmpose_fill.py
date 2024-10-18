"""
Goes through a directory of images and their annotations in dataframe format and runs 
pose estimation for birds on each cropped image. 
Saves each image with its keypoints and skeleton drawn on it.
Finally, returns the updated dataframe (with the skeleton information).
"""

import os, ast
from mmpose.apis import MMPoseInferencer
import cv2
import pandas as pd
import numpy as np


def fill_keypoints(df, image_dir="", cache_path=None):

    # Initialize the MMPoseInferencer 
    inferencer = MMPoseInferencer(
        pose2d='td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256',
        pose2d_weights='https://download.openmmlab.com/mmpose/v1/animal_2d_keypoint/topdown_heatmap/animal_kingdom/td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256-566feff5_20230519.pth',
        device='cpu',
    )

    if 'keypoints' not in df.columns:
        df['keypoints'] = None  # Initialize with None or an empty list
    if 'num_keypoints' not in df.columns:
        df['num_keypoints'] = 0  # Initialize with zero

    # Load or initialize the cache DataFrame
    if cache_path and os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        # Ensure 'keypoints' and 'num_keypoints' columns exist in cache_df
        if 'keypoints' not in cache_df.columns:
            cache_df['keypoints'] = None
        if 'num_keypoints' not in cache_df.columns:
            cache_df['num_keypoints'] = 0
    else:
        cache_df = pd.DataFrame(columns=df.columns)

    # Loop through each row and update with keypoints
    for idx, row in df.iterrows():

        if cache_path and 'path' in cache_df.columns:
            cached_row = cache_df[cache_df['path'] == row['path']]
            if not cached_row.empty and pd.notnull(cached_row.iloc[0]['keypoints']):
                # Use cached data: update only the keypoints and num_keypoints columns
                df.at[idx, 'keypoints'] = cached_row.iloc[0]['keypoints']
                df.at[idx, 'num_keypoints'] = cached_row.iloc[0]['num_keypoints']
                continue

        # else:

        image_id = int(row['image_id'])
        bbox = row['bbox']  # [x1, y1, width, height]
        if type(bbox) == str:
            bbox = ast.literal_eval(bbox)
        img_path = os.path.join(image_dir, row['path'])

        if not os.path.exists(img_path):
            print(f"File does not exist: {img_path}")
            continue

        img = cv2.imread(img_path)

        # Crop the image using the bounding box
        x, y, w, h = [int(v) for v in bbox]

        cropped_img = img[y:y+h, x:x+w]
        result_generator = inferencer(cropped_img)

        # Perform pose estimation using the bounding box from the annotation
        result = next(result_generator)

        # Retrieve and process heatmaps

        formatted_keypoints = []
        keypoint_scores = []
        
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
        df.at[idx, 'keypoints'] = str(formatted_keypoints)
        df.at[idx, 'num_keypoints'] = len(keypoint_scores)

        if cache_path:
            cache_index = cache_df.index[cache_df['path'] == row['path']].tolist()
            if cache_index:
                cache_idx = cache_index[0]
                cache_df.at[cache_idx, 'keypoints'] = formatted_keypoints
                cache_df.at[cache_idx, 'num_keypoints'] = len(keypoint_scores)

        print(f"Processed image: {image_id} with number of keypoints {len(keypoint_scores)}")

    if cache_path:
        cache_df.to_csv(cache_path, index=False)

    return df

def get_skeleton_info():
    return [
                [2,1],[3,1],[4,5],[4,6],[7,5],[7,6],[1,14],[14,21],
                [21,22],[22,23],[1,8],[1,9],[8,10],[9,11],[10,12],[11,13],
                [21,15],[21,16],[15,17],[16,18],[17,19],[18,20]
            ]

def get_keypoints_info():
    return ["Head_Mid_Top","Eye_Left","Eye_Right","Mouth_Front_Top","Mouth_Back_Left",
            "Mouth_Back_Right","Mouth_Front_Bottom","Shoulder_Left","Shoulder_Right",
            "Elbow_Left","Elbow_Right","Wrist_Left","Wrist_Right","Torso_Mid_Back",
            "Hip_Left","Hip_Right","Knee_Left","Knee_Right","Ankle_Left","Ankle_Right",
            "Tail_Top_Back","Tail_Mid_Back","Tail_End_Back"]