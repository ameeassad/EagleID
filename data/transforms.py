import random
import math
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

def rotate_image(image, angle):
    # Convert PIL image to NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
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


class ResizeAndPadRGBSkeleton:
    def __init__(self, size):
        self.size = size

        self.resize_and_pad_rgb = ResizeAndPadRGB(size)
        self.resize_and_pad_skeleton = ResizeAndPadSkeleton(size)


    def __call__(self, image, skeleton):
        rgb = self.resize_and_pad_rgb(image)
        new_width, new_height, pad_width, pad_height = self.resize_and_pad_rgb.get_img_parameters(image)
        skeleton = self.resize_and_pad_skeleton(skeleton, new_width, new_height, pad_width, pad_height)
        return rgb, skeleton



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
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15, color_and_gaussian=True):
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
        if random.random() < 0.5:
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
        skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Concatenate RGB and skeleton channels
        concatenated = torch.cat((rgb_img, skeleton_channel), dim=0)

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
        if random.random() < 0.5:
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

        if self.color_and_gaussian:
            #  Apply the transform to rgb_img and record the replay data
            augmented = self.color_transform(image=rgb_img)
            rgb_img = augmented['image']
            replay_data = augmented['replay']

            # Apply the recorded replay data to all cropped images
            for component_name in cropped_images:
                if cropped_images[component_name] is not None:
                    cropped_images[component_name] = A.ReplayCompose.replay(replay_data, image=cropped_images[component_name])['image']

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
                component_tensor = torch.tensor(cropped_img, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
                component_tensors.append(component_tensor)

        # Concatenate the original image tensor with all component tensors along the channel dimension
        target_size = (rgb_img_tensor.shape[1], rgb_img_tensor.shape[2])  # (height, width)
        # Use interpolate to resize tensors
        resized_component_tensors = [
            torch.nn.functional.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            for tensor in component_tensors
        ]
        # Concatenate tensors along channel dimension
        concatenated = torch.cat([rgb_img_tensor] + resized_component_tensors, dim=0)  # [C_total, H, W]

        # Apply normalization to the concatenated tensor entirely
        concatenated = concatenated / 255.0  # Scale to [0, 1]
        for c in range(concatenated.shape[0]):
            concatenated[c] = (concatenated[c] - self.mean[c % 3]) / self.std[c % 3]

        return concatenated
    

class RGBTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15, color_and_gaussian=True):

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
        # return self.transforms(img)

        # Flip RGB (NumPy array)
        if random.random() < 0.5:
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


# def denorm_RGB_components(image, mean, std):
#     # image: Tensor of shape (C, H, W) or (H, W, C)
#     # mean and std should be lists of length 3 to represent RGB mean and std.
    
#     # Denormalize each channel
#     for c in range(image.shape[0]):
#         image[c] = (image[c] * std[c % 3]) + mean[c % 3]
    
#     # Scale back to [0, 255]
#     image = image * 255.0
#     return image

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