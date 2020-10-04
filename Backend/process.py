import cv2
import torch
import numpy as np

def preprocess_mri(image):
    image = cv2.resize(image, (256, 256))
    image = torch.tensor(image.astype(np.float32))
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)

    return image

def postprocess_mask(mask):
    mask = mask.detach()
    mask = mask.cpu()
    mask = mask.numpy()[0, 0, :, :] # grayscale to rgb

    return mask
