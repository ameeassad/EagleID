import random
import math
import ast
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
    RandomErasing,
)
import torchvision.transforms.functional as F
import torch.nn.functional
from torchvision.transforms.functional import resize, pad


# Coloration and blurring transforms
def get_some_transforms():
    return A.SomeOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.GaussNoise(var_limit=(0.0, 0.01 * 255), p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], 
        n=random.randint(0, 2), 
        # n=2,
        p=1.0
    )

def get_color_transforms():
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], p=1.0)

def get_blur_transforms():
    return A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(0.0, 0.01 * 255), p=1.0),
        ], p=0.5)

# Advanced augmentations for transformer models
def get_advanced_transforms():
    """Advanced augmentations specifically for transformer models"""
    return A.Compose([
        # AutoAugment-style policies
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.5),
        
        # Color augmentations
        A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        # Geometric augmentations
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ], p=0.3),
    ], p=1.0)

def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rotate_image(image, angle):    

    # Convert PIL image to NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Check if the image is entirely black
    if np.sum(image) == 0:
        # Return the black image as is, since rotating an all-black image will still be black
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated_image

def resize_and_pad(image, size, skeleton_channel=None):
    """
    Resizes and pads both the RGB image and the skeleton channel based on the RGB image dimensions.

    Args:
        image (PIL.Image): RGB image to be resized and padded.
        size (int): Target size for resizing.
        skeleton_channel (np.array): Skeleton channel to be resized and padded.

    Returns:
        tuple: Resized and padded RGB image and skeleton channel.
    """

    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if original_width > original_height:
        new_width = size
        new_height = math.ceil(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = math.ceil(new_height * aspect_ratio)

    resized_image = resize(image, (new_height, new_width))

    # Calculate padding
    pad_width = size - new_width
    pad_height = size - new_height

    padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
    
    # Pad the image to make it square
    padded_image = pad(resized_image, padding, fill=0, padding_mode='constant')

    # Padding for skeleton channel (using np.pad)
    if skeleton_channel is not None and skeleton_channel.size > 0:
        skeleton_channel_resized = cv2.resize(skeleton_channel, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        padding_skeleton = ((pad_height // 2, pad_height - (pad_height // 2)),
                            (pad_width // 2, pad_width - (pad_width // 2)))
        padded_skeleton_channel = np.pad(skeleton_channel_resized, padding_skeleton, mode='constant', constant_values=0)

    else:
        padded_skeleton_channel = np.zeros((size, size), dtype=np.float32)

    if skeleton_channel is not None:
        return padded_image, padded_skeleton_channel
    else: 
        return padded_image


class TransformerRGBTransforms:
    """
    Advanced transforms for transformer models with MixUp, CutMix, Random Erasing, and AutoAugment-style augmentations.
    Designed to work with the resize_and_pad pipeline.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                 mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=0.0, cutmix_prob=0.0,
                 random_erasing_prob=0.3, advanced_aug_prob=0.0):
        self.mean = mean
        self.std = std
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob  # Start with 0 probability
        self.cutmix_prob = cutmix_prob  # Start with 0 probability
        self.random_erasing_prob = random_erasing_prob
        self.advanced_aug_prob = advanced_aug_prob  # Start with 0 probability
        self.use_advanced_aug = False  # Flag to control advanced augmentations
        
        # Advanced augmentations
        self.advanced_transforms = get_advanced_transforms()
        
        # Basic transforms
        self.basic_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Transpose(p=0.3),
        ])
        
        # Normalization
        self.normalize_transform = A.Normalize(mean=mean, std=std)
        
        # Random Erasing (applied after tensor conversion)
        self.random_erasing = RandomErasing(
            p=random_erasing_prob, 
            scale=(0.02, 0.33), 
            ratio=(0.3, 3.3), 
            value=0
        )

    def __call__(self, rgb_img):
        """
        Apply transforms to the image.
        Note: This is called AFTER resize_and_pad, so the image is already square and padded.
        """
        # Convert PIL to numpy if needed
        if isinstance(rgb_img, Image.Image):
            rgb_img = np.array(rgb_img)
        
        # Apply basic geometric transforms
        if random.random() < 0.8:
            rgb_img = self.basic_transforms(image=rgb_img)['image']
        
        # Apply advanced augmentations only if enabled
        if self.use_advanced_aug and random.random() < self.advanced_aug_prob:
            rgb_img = self.advanced_transforms(image=rgb_img)['image']
        
        # Normalization
        rgb_img = self.normalize_transform(image=rgb_img)['image']
        
        # Convert to Tensor
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        
        # Apply Random Erasing
        if random.random() < self.random_erasing_prob:
            rgb_img = self.random_erasing(rgb_img)
        
        return rgb_img

    def apply_mixup_cutmix(self, batch_x, batch_y):
        """
        Apply MixUp or CutMix to a batch of data.
        This should be called in the training step, not in the transform.
        """
        if random.random() < self.mixup_prob:
            batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, self.mixup_alpha)
            return batch_x, y_a, y_b, lam, 'mixup'
        elif random.random() < self.cutmix_prob:
            batch_x, y_a, y_b, lam = cutmix_data(batch_x, batch_y, self.cutmix_alpha)
            return batch_x, y_a, y_b, lam, 'cutmix'
        else:
            return batch_x, batch_y, batch_y, 1.0, 'none'


class ResizeAndPadBoth:
    def __init__(self, size, skeleton=True):
        self.size = size

        self.resize_and_pad_rgb = ResizeAndPadRGB(size)
        if skeleton:
            self.resize_and_pad_skeleton = ResizeAndPadSkeleton(size)
        else:  # Heatmaps
            self.resize_and_pad_skeleton = ResizeAndPadHeatmaps(size)


    def __call__(self, image, skeleton_or_heatmaps):
        rgb = self.resize_and_pad_rgb(image)
        new_width, new_height, pad_width, pad_height = self.resize_and_pad_rgb.get_img_parameters(image)
        skeleton_or_heatmaps = self.resize_and_pad_skeleton(skeleton_or_heatmaps, new_width, new_height, pad_width, pad_height)
        return rgb, skeleton_or_heatmaps



class ResizeAndPadRGB:
    def __init__(self, size):
        self.size = size

    def get_img_parameters(self, image):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = self.size
            new_height = math.ceil(new_width / aspect_ratio)
        else:
            new_height = self.size
            new_width = math.ceil(new_height * aspect_ratio)

        # Calculate padding
        pad_width = self.size - new_width
        pad_height = self.size - new_height

        return new_width, new_height, pad_width, pad_height


    def __call__(self, image):

        new_width, new_height, pad_width, pad_height = self.get_img_parameters(image)

        resized_image = resize(image, (new_height, new_width))

        padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
        
        # Pad the image to make it square
        padded_image = pad(resized_image, padding, fill=0, padding_mode='constant')
        
        return padded_image
    
class ResizeAndPadSkeleton:
    def __init__(self, size):
        self.size = size

    def __call__(self, skeleton_channel, new_width, new_height, pad_width, pad_height):

        # Padding for skeleton channel (using np.pad)
        if skeleton_channel is not None and skeleton_channel.size > 0:
            skeleton_channel_resized = cv2.resize(skeleton_channel, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            padding_skeleton = ((pad_height // 2, pad_height - (pad_height // 2)),
                                (pad_width // 2, pad_width - (pad_width // 2)))
            padded_skeleton_channel = np.pad(skeleton_channel_resized, padding_skeleton, mode='constant', constant_values=0)

        else:
            padded_skeleton_channel = np.zeros((self.size, self.size), dtype=np.float32)

        return padded_skeleton_channel
    
class ResizeAndPadHeatmaps:
    def __init__(self, size):
        self.size = size

    def __call__(self, heatmap_channels, new_width, new_height, pad_width, pad_height):
        # Initialize a list to store resized and padded heatmaps
        padded_heatmaps = []

        # Process each heatmap channel independently
        for heatmap in heatmap_channels:
            if heatmap is not None and heatmap.size > 0:
                # Resize the heatmap to the new dimensions
                heatmap_resized = cv2.resize(heatmap, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Pad the resized heatmap to the desired final size
                padding = ((pad_height // 2, pad_height - (pad_height // 2)),
                           (pad_width // 2, pad_width - (pad_width // 2)))
                padded_heatmap = np.pad(heatmap_resized, padding, mode='constant', constant_values=0)

                padded_heatmaps.append(padded_heatmap)
            else:
                # If heatmap is invalid, append a zero array with the target size
                padded_heatmaps.append(np.zeros((self.size, self.size), dtype=np.float32))

        # Stack all heatmaps into a single array with multiple channels
        padded_heatmaps = np.stack(padded_heatmaps, axis=0)  # Shape: (num_channels, size, size)

        return padded_heatmaps

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
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=45, color_and_gaussian=True):
        self.mean = mean
        self.std = std
        self.degrees = degrees
        self.color_and_gaussian = color_and_gaussian
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

        if random.random() < 0.5 and self.degrees !=0: # Don't flip if validation
            rgb_img = np.fliplr(rgb_img)  # Flip RGB (NumPy array)
            skeleton_channel = np.fliplr(skeleton_channel)  # Flip skeleton (NumPy array)

        # Apply the same random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        # rgb_img = rgb_img.rotate(angle)
        # skeleton_channel = self.rotate_image(skeleton_channel, angle)
        rgb_img = rotate_image(rgb_img, angle)  # Rotate RGB (NumPy array)
        skeleton_channel = rotate_image(skeleton_channel, angle) 

        # Apply some of the transforms to the RGB image
        if self.color_and_gaussian:
            rgb_img = self.some_of_transforms(image=rgb_img)['image']

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
        skeleton_channel = torch.tensor(skeleton_channel.copy(), dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Concatenate RGB and skeleton channels
        concatenated = torch.cat((rgb_img, skeleton_channel), dim=0)

        return concatenated
    
class SynchMultiChannelTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15, color_and_gaussian=True):
        self.mean = mean
        self.std = std
        self.degrees = degrees
        self.color_and_gaussian = color_and_gaussian
        self.some_of_transforms = get_some_transforms()

        self.normalize_transform = A.Normalize(mean=mean, std=std)

        
    def __call__(self, rgb_img, heatmap_channels=None):
        """
        Args:
            rgb_img (np.array): The RGB image (H, W, 3).
            heatmap_channels (list or np.array): List of heatmap channels (each H, W) or array with shape (H, W, num_keypoints).
        Returns:
            concatenated (torch.Tensor): Concatenated tensor of RGB and heatmap channels (C, H, W).
        """

        # Apply the same random horizontal flip
        if random.random() < 0.5 and self.degrees !=0:
            rgb_img = np.fliplr(rgb_img)  # Flip RGB (NumPy array)
            heatmap_channels = [np.fliplr(hm) for hm in heatmap_channels]  # Flip all heatmap channels

        # Apply the same random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        rgb_img = rotate_image(rgb_img, angle)  # Rotate RGB (NumPy array)
        heatmap_channels = [rotate_image(hm, angle) for hm in heatmap_channels]  # Rotate all heatmap channels

        # Apply some of the color transforms to the RGB image
        if self.color_and_gaussian:
            rgb_img = self.some_of_transforms(image=rgb_img)['image']

        # Apply normalization to the RGB image
        rgb_img = self.normalize_transform(image=rgb_img)['image']

        # # Ensure heatmap channels have the same spatial dimensions as the rgb_img.
        # if heatmap_channels is not None:
        #     rgb_h, rgb_w, _ = rgb_img.shape
        #     new_heatmaps = []
        #     for hm in heatmap_channels:
        #         # Check if current heatmap's dimensions match the RGB image.
        #         hm_h, hm_w = hm.shape[:2]
        #         if (hm_h, hm_w) != (rgb_h, rgb_w):
        #             # Resize heatmap to match: note that cv2.resize expects (width, height)
        #             hm_resized = cv2.resize(hm, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
        #         else:
        #             hm_resized = hm
        #         new_heatmaps.append(hm_resized)
        #     heatmap_channels = new_heatmaps
        
        # Convert both RGB image and heatmap channels to tensors
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W] for RGB
        heatmap_channels = [torch.tensor(hm.copy(), dtype=torch.float32).unsqueeze(0) for hm in heatmap_channels]  # [1, H, W] for each heatmap

        # Concatenate all heatmap channels together along the channel dimension
        heatmap_tensor = torch.cat(heatmap_channels, dim=0)  # [num_keypoints, H, W]

        # Concatenate RGB and heatmap channels together along the channel dimension
        concatenated = torch.cat((rgb_img, heatmap_tensor), dim=0)  # [3 + num_keypoints, H, W]

        return concatenated
    

class ComponentGenerationTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15, color_and_gaussian=True):
        self.mean = mean
        self.std = std
        self.degrees = degrees
        self.color_and_gaussian = color_and_gaussian
        self.color_transform = A.ReplayCompose([get_color_transforms()])
        self.gaussian_transform = get_blur_transforms()

    def __call__(self, rgb_img, cropped_images):
        # Apply the same random horizontal flip
        if random.random() < 0.5 and self.degrees !=0:
            rgb_img = np.fliplr(rgb_img)  # Flip RGB (NumPy array)
            for component_name in cropped_images:
                if cropped_images[component_name] is not None:
                    cropped_images[component_name] = np.fliplr(cropped_images[component_name])
                else:
                    # Create a blank black image of the same height and width as rgb_img
                    height, width, _ = rgb_img.shape
                    cropped_images[component_name] = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply the same random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        rgb_img = rotate_image(rgb_img, angle)  # Rotate RGB (NumPy array)
        for component_name in cropped_images:
            if cropped_images[component_name] is not None:
                cropped_images[component_name] = rotate_image(cropped_images[component_name], angle)
            else:
                # Create a blank black image of the same height and width as rgb_img
                height, width, _ = rgb_img.shape
                cropped_images[component_name] = np.zeros((height, width, 3), dtype=np.uint8)

        if self.color_and_gaussian:
            #  Apply the transform to rgb_img and record the replay data
            augmented = self.color_transform(image=rgb_img)
            rgb_img = augmented['image']
            replay_data = augmented['replay']

            # Apply the recorded replay data to all cropped images
            for component_name in cropped_images:
                if cropped_images[component_name] is not None and cropped_images[component_name].shape[0] > 0 and cropped_images[component_name].shape[1] > 0:
                    try:
                        cropped_images[component_name] = A.ReplayCompose.replay(replay_data, image=cropped_images[component_name])['image']
                    except Exception as e:
                        # Handle any exceptions by replacing the invalid image with a fallback black image
                        print(f"Warning: Error applying replay to component '{component_name}', using a black image. Error: {e}")
                        cropped_images[component_name] = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    # Ensure cropped images are not None
                    height, width, _ = rgb_img.shape
                    cropped_images[component_name] = np.zeros((height, width, 3), dtype=np.uint8)

            # only apply gaussian to rgb_img
            rgb_img = self.gaussian_transform(image=rgb_img)['image']
            
        # # Apply normalization to each idividually 
        # rgb_img = self.normalize_transform(image=rgb_img)['image']
        # for component_name in cropped_images:
        #     if cropped_images[component_name] is not None:
        #         cropped_images[component_name] = self.normalize_transform(image=cropped_images[component_name])['image']

        # Convert the RGB image to a tensor
        rgb_img_tensor = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]

        # Convert all cropped images to tensors and concatenate with the original image
        component_tensors = []
        for component_name, cropped_img in cropped_images.items():
            if cropped_img is not None:
                component_tensor = torch.tensor(cropped_img.copy(), dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
                component_tensors.append(component_tensor)

        # Concatenate the original image tensor with all component tensors along the channel dimension
        target_size = (rgb_img_tensor.shape[1], rgb_img_tensor.shape[2])  # (height, width)
        # # Use interpolate to resize tensors
        # resized_component_tensors = [
        #     torch.nn.functional.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        #     for tensor in component_tensors
        # ]
        resized_component_tensors = []
        for tensor in component_tensors:
            if tensor.size(1) > 0 and tensor.size(2) > 0:
                # Interpolate to resize tensors
                resized_tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                # Handle invalid dimensions by creating a fallback black tensor
                resized_tensor = torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)
                print("Warning: Component tensor has invalid dimensions, creating a fallback black tensor.")
            resized_component_tensors.append(resized_tensor)

        # Concatenate tensors along channel dimension
        concatenated = torch.cat([rgb_img_tensor] + resized_component_tensors, dim=0)  # [C_total, H, W]

        # Apply normalization to the concatenated tensor entirely
        concatenated = concatenated / 255.0  # Scale to [0, 1]
        for c in range(concatenated.shape[0]):
            concatenated[c] = (concatenated[c] - self.mean[c % 3]) / self.std[c % 3]

        return concatenated
    

class RGBTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=45, color_and_gaussian=True):

        self.mean = mean
        self.std = std
        self.degrees = degrees
        self.some_of_transforms = get_some_transforms()
        self.normalize_transform = A.Normalize(mean=mean, std=std)
        self.color_and_gaussian = color_and_gaussian

        # self.transforms = Compose([
        #     RandomHorizontalFlip(),
        #     RandomRotation(degrees=degrees),
        #     ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        #     ToTensor(),
        #     Normalize(mean=mean, std=std),
        # ])
    def __call__(self, rgb_img):
        # Flip RGB (NumPy array)
        if random.random() < 0.5 and self.degrees !=0:
            rgb_img = np.fliplr(rgb_img)

        # Random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        rgb_img = rotate_image(rgb_img, angle) 

        # Coloration transforms
        if self.color_and_gaussian:
            rgb_img = self.some_of_transforms(image=rgb_img)['image']

        # Normalization
        rgb_img = self.normalize_transform(image=rgb_img)['image']

        # Convert to Tensor
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]

        return rgb_img

def denormalize(x, mean, std):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid format for mean or std: {x}")

    #  x is np array
    x = x[:3, :, :]

    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)

    denorm = A.Normalize( 
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        always_apply=True,
        max_pixel_value=1.0
    )
    x = (denorm(image=x)["image"]*255).astype(np.uint8) 

    return x


def denorm_RGB_components(image, mean, std):
    """
    Denormalizes an image tensor that includes RGB and component channels.
    
    Args:
        image (numpy.ndarray): Tensor of shape (C, H, W), where C represents multiple channels (e.g., RGB + components).
        mean (list): List of mean values for RGB channels.
        std (list): List of standard deviation values for RGB channels.

    Returns:
        numpy.ndarray: Denormalized image, scaled back to [0, 255].
    """
    # Denormalize each channel based on repeating RGB mean and std
    denormalized_image = np.empty_like(image, dtype=np.float32)
    
    for c in range(image.shape[0]):
        denormalized_image[c] = (image[c] * std[c % 3]) + mean[c % 3]
    
    # Scale back to [0, 255]
    denormalized_image = np.clip(denormalized_image * 255.0, 0, 255)  # Clip to avoid values > 255 or < 0
    return denormalized_image