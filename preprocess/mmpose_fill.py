"""
Goes through a directory of images and their annotations in dataframe format and runs 
pose estimation for birds on each cropped image. 
Saves each image with its keypoints and skeleton drawn on it.
Finally, returns the updated dataframe (with the skeleton information).
"""

import os, ast
import cv2
import pandas as pd
import numpy as np


def fill_keypoints(df, image_dir="", cache_path=None, only_cache=False, device='cpu', animal_cat='bird'):

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
    
    if only_cache:
        # Loop through each row and update with keypoints
        for idx, row in df.iterrows():
            if cache_path and 'path' in cache_df.columns:
                cached_row = cache_df[cache_df['path'] == row['path']]
                if not cached_row.empty and pd.notnull(cached_row.iloc[0]['keypoints']):
                    # Use cached data: update only the keypoints and num_keypoints columns
                    df.at[idx, 'keypoints'] = cached_row.iloc[0]['keypoints']
                    df.at[idx, 'num_keypoints'] = cached_row.iloc[0]['num_keypoints'] 
        return df # return without doing any inference with mmpose
        
    from mmpose.apis import MMPoseInferencer

    pose2d = f'td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_{animal_cat}-256x256'

    if animal_cat == 'bird':
        download_id = '-566feff5'
    elif animal_cat == 'mammal':
        download_id = '-e8aadf02'
    elif animal_cat == 'amphibian':
        download_id = '-845085f9'
    elif animal_cat == 'reptile':
        download_id = '-e8440c16'
    elif animal_cat == 'fish':
        download_id = '-76c3999f'
    else:
        print(f"Invalid animal category: {animal_cat}, using default top-down model.")
        pose2d = 'td-hm_hrnet-w32_8xb32-300e_animalkingdom_P1-256x256'
        download_id - '-08bf96cb'
    
    pose2d_weights = f'https://download.openmmlab.com/mmpose/v1/animal_2d_keypoint/topdown_heatmap/animal_kingdom/{pose2d}{download_id}_20230519.pth'

    # Initialize the MMPoseInferencer 
    inferencer = MMPoseInferencer(
        pose2d=pose2d,
        pose2d_weights=pose2d_weights,
        device=device,
    )

    # Loop through each row and update with keypoints
    for idx, row in df.iterrows():
        
        if cache_path and 'path' in cache_df.columns:
            cached_row = cache_df[cache_df['path'] == row['path']]
            if (
                not cached_row.empty
                and cached_row.iloc[0]['keypoints'] is not None
                and isinstance(cached_row.iloc[0]['keypoints'], (list, np.ndarray))  # Ensure it's list-like
                and all(pd.notnull(value) for value in cached_row.iloc[0]['keypoints'])
                and len(cached_row.iloc[0]['keypoints']) == 23
                ):
                # Use cached data: update only the keypoints and num_keypoints columns
                df.at[idx, 'keypoints'] = cached_row.iloc[0]['keypoints']
                df.at[idx, 'num_keypoints'] = cached_row.iloc[0]['num_keypoints']
                continue

        # else:
        image_id = row['image_id']
        # image_id = int(row['image_id'])
        bbox = row['bbox']  # [x1, y1, width, height]
        if type(bbox) == str:
            bbox = ast.literal_eval(bbox)
        img_path = os.path.join(image_dir, row['path'])

        if not os.path.exists(img_path):
            print(f"File does not exist: {img_path}")
            continue

        img = cv2.imread(img_path)

        # Crop the image using the bounding box
        if not (type(bbox)!=list and bbox.isna()):
            x, y, w, h = [int(v) for v in bbox]
            img = img[y:y+h, x:x+w]

        result_generator = inferencer(img)

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
    """"
    Returns the list of keypoints for any Animal Kingdom dataset for pose estimation.
    For more info on keypoint structure see:
    https://github.com/sutdcv/Animal-Kingdom/blob/master/Animal_Kingdom/pose_estimation/code/code_new/mmpose/mmpose/datasets/datasets/animal/animal_ak_dataset.py
    """
    return ["Head_Mid_Top","Eye_Left","Eye_Right","Mouth_Front_Top","Mouth_Back_Left",
            "Mouth_Back_Right","Mouth_Front_Bottom","Shoulder_Left","Shoulder_Right",
            "Elbow_Left","Elbow_Right","Wrist_Left","Wrist_Right","Torso_Mid_Back",
            "Hip_Left","Hip_Right","Knee_Left","Knee_Right","Ankle_Left","Ankle_Right",
            "Tail_Top_Back","Tail_Mid_Back","Tail_End_Back"]