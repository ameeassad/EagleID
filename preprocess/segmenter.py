import ultralytics
from ultralytics import YOLO, utils

import cv2
import os
from PIL import Image
import numpy as np

import matplotlib as plt
import pandas as pd
from ultralytics import YOLO

from shapely.geometry import Polygon


def get_segs(self, polygon, image):
    segmentation = [polygon]

    poly = Polygon(polygon)
    area = poly.area
    min_x, min_y, max_x, max_y = poly.bounds
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

    seg = polygon.flatten().tolist()
    # [x, y, w, h] = cv2.boundingRect(polygon)
    return seg, bbox, area

def display_annotations(self,image,polygon=None):
    # Create a subplot
    fig, ax = plt.pyplot.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    if polygon is not None:
        # # Display the masked image
        # ax[1].imshow(mask_img)
        # ax[1].set_title('Segmentation Mask')
        # ax[1].axis('off')

        # Convert polygon points to integer
        polygon_points = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)

        image_np_copy = np.array(image).copy()
        cv2.polylines(image_np_copy, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

        ax[1].imshow(image_np_copy)
        ax[1].set_title('Polygon')
        ax[1].axis('off')

        # Show the plot
        plt.pyplot.show()


def add_segmentations(df, image_dir="", testing=False):

    model = YOLO('../checkpoints/yolov8x-seg.pt')

    updated_rows = []
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row['path'])
        image = Image.open(image_path)
        W, H = image.size

        results = model(image)
        for result in results:
            if result.masks is None:
                print("No mask found in result, printing result")
                # display_annotations(image)
                continue
            for mask in result.masks:
                polygon = mask.xy[0]

                # FOR VISUALISATION - uncomment
                # display_annotations(image, polygon)

                segmentation, bbox, area = get_segs(polygon, image)
                iscrowd = 0 if len(results) <= 1 else 1

                updated_row = row.to_dict()
                updated_row['segmentations'] = [segmentation]
                updated_row['height'] = H
                updated_row['width'] = W
                updated_row['bbox'] = bbox
                updated_row['area'] = int(area)
                updated_row['iscrowd'] = iscrowd

                updated_rows.append(updated_row)

    updated_df = pd.DataFrame(updated_rows)
    return updated_df


def preprocess_lvl2(df, image_dir, yolo_model, testing=False):
    model = YOLO(yolo_model)
    # Process images and update the DataFrame with segmentation and size information
    df = add_segmentations(df, image_dir, model)
    print("Updated DataFrame with segmentations.")

    return df