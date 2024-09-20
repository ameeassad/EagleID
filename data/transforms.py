import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    Pad,
    RandomRotation,
    ColorJitter,
)
import torchvision.transforms.functional as F
from torchvision.transforms.functional import resize, pad

# Coloration and blurring transforms
def get_some_transforms():
    return A.SomeOf([
        A.GaussianBlur(blur_limit=(0, 1.5), p=0.5),
        A.GaussNoise(var_limit=(0.0, 0.01 * 255), p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], n=random.randint(0, 2), p=1.0)

def rotate_image(image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        return rotated_image


class ValTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), skeleton=False):
        self.skeleton = skeleton
        self.rgb_transforms = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, skeleton_channel=None):
        # Apply basic transformations to the image (RGB)
        image = self.rgb_transforms(image)

        # Apply to tensor to the skeleton channel
        skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32)
        
        skeleton_channel = skeleton_channel.unsqueeze(0)
        concatenated = torch.cat((image, skeleton_channel), dim=0)
     
        return concatenated
    
    
# Uses albumentations
class SynchTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15):
        self.mean = mean
        self.std = std
        self.degrees = degrees
        self.some_of_transforms = get_some_transforms()

        self.normalize_transform = A.Normalize(mean=mean, std=std)

        # self.rgb_transforms = Compose([
        #     # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        #     ToTensor(),
        #     Normalize(mean=mean, std=std),
        # ])
        
    def __call__(self, rgb_img, skeleton_channel=None):
        # Apply the same random horizontal flip
        # if random.random() < 0.5:
        #     rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip RGB
        #     skeleton_channel = cv2.flip(skeleton_channel, 1)  # Flip skeleton
        if random.random() < 0.5:
            rgb_img = np.fliplr(rgb_img)  # Flip RGB (NumPy array)
            skeleton_channel = np.fliplr(skeleton_channel)  # Flip skeleton (NumPy array)

        # Apply the same random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        # rgb_img = rgb_img.rotate(angle)
        # skeleton_channel = self.rotate_image(skeleton_channel, angle)
        rgb_img = rotate_image(rgb_img, angle)  # Rotate RGB (NumPy array)
        skeleton_channel = rotate_image(skeleton_channel, angle) 

        # Apply some of the transforms to the RGB image and the skeleton channel
        rgb_img = self.some_of_transforms(image=rgb_img)['image']
        skeleton_channel = self.some_of_transforms(image=skeleton_channel)['image']

        # # Apply additional transforms to the RGB image
        # rgb_img = self.rgb_transforms(rgb_img)
        # Apply normalization to the RGB image
        rgb_img = self.normalize_transform(image=rgb_img)['image']

        # Convert the skeleton channel to tensor after transformations
        # skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension [1, H, W]
        # # skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32)
        # # skeleton_channel = skeleton_channel.unsqueeze(0)
        
        # Convert both channels to tensors
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Concatenate RGB and skeleton channels
        concatenated = torch.cat((rgb_img, skeleton_channel), dim=0)

        return concatenated

class RGBTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15):

        self.mean = mean
        self.std = std
        self.degrees = degrees
        self.some_of_transforms = get_some_transforms()
        self.normalize_transform = A.Normalize(mean=mean, std=std)

        # self.transforms = Compose([
        #     RandomHorizontalFlip(),
        #     RandomRotation(degrees=degrees),
        #     ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        #     ToTensor(),
        #     Normalize(mean=mean, std=std),
        # ])
    def __call__(self, img):
        # return self.transforms(img)

        # Flip RGB (NumPy array)
        if random.random() < 0.5:
            rgb_img = np.fliplr(img)

        # Random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        rgb_img = rotate_image(rgb_img, angle) 

        # Coloration transforms
        rgb_img = self.some_of_transforms(image=rgb_img)['image']

        # Normalization
        rgb_img = self.normalize_transform(image=rgb_img)['image']

        # Convert to Tensor
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]

        return rgb_img

