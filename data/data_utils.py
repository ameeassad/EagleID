import torch
from preprocess.mmpose_fill import get_keypoints_info

def unnormalize(x, mean, std):
    """
    Unnormalizes a tensor by applying the inverse of the normalization transform.

    Args:
        x (torch.Tensor): Tensor to be unnormalized.
        mean (tuple): Mean used for normalization.
        std (tuple): Standard deviation used for normalization.

    Returns:
        torch.Tensor: Unnormalized tensor.
    """
    
    x = x.clone().detach()[:3]

    mean = (mean, mean, mean) if isinstance(mean, float) else tuple(mean)
    std = (std, std, std) if isinstance(std, float) else tuple(std)

    mean = torch.tensor(mean)
    std = torch.tensor(std)


    if x.dim() == 3:  # Ensure the tensor has 3 dimensions
        unnormalized_x = x.clone()
        for t, m, s in zip(unnormalized_x, mean, std):
            t.mul_(s).add_(m)
        return unnormalized_x
    else:
        raise ValueError(f"Expected input tensor to have 3 dimensions, but got {x.dim()} dimensions.")
    

def calculate_num_channels(preprocess_lvl):
    img_channels = 3 
    if preprocess_lvl == 0:
        method = 'full'
    if preprocess_lvl == 1:
        method = 'bbox'
    elif preprocess_lvl == 2: 
        method = "bbox_mask"
    elif preprocess_lvl == 3: 
        method = "bbox_mask_skeleton"
        img_channels = 4
    elif preprocess_lvl == 4:
        method = "bbox_mask_components"
        img_channels = 3 + 3 * 5
    elif preprocess_lvl == 5:
        method = "bbox_mask_heatmaps"
        img_channels = 3 + len(get_keypoints_info())
    
    return img_channels