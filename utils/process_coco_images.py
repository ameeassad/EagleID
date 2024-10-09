"""
base code https://github.com/owahltinez/triplet-loss-animal-reid/blob/main/process_coco_images.py

Process image folders and annotations from COCO format to filesystem-based format for training.
"""
import json
from pathlib import Path

from PIL import Image

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process image and annotation paths.")
    
    # Define the command-line arguments
    parser.add_argument("--images", type=str, required=True, help="Path of images folder.")
    parser.add_argument("--annotations", type=str, required=True, help="Path of annotations file.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory where images will be saved.")
    
    # Parse the arguments
    return parser.parse_args()

def main(args):
  images_path = args.images
  outdir_path = args.outdir

  with open(args.annotations) as fh:
    metadata = json.load(fh)

  images = {x["id"]: images_path / x["file_name"] for x in metadata["images"]}
  for annotation in metadata["annotations"]:
    l, t, w, h = annotation["bbox"]
    individual_id = annotation["name"]
    img = Image.open(images[annotation["image_id"]])
    img = img.crop((l, t, l + w, t + h))
    (outdir_path / individual_id).mkdir(parents=True, exist_ok=True)
    img.save(outdir_path / individual_id / (annotation["uuid"] + ".jpg"))


if __name__ == "__main__":
    args = parse_args()
    main(args)