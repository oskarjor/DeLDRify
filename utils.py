import cv2 as cv
import numpy as np
import PIL.Image
import torch

# functions for converting between numpy arrays representing HDR images and PIL images
def hdr_np_to_img_manual(hdr_array: np.ndarray) -> PIL.Image.Image:
    """Converts a numpy array representing an HDR image to a PIL image.

    Args:
        hdr_array (np.ndarray): Numpy array representing an HDR image. Values are in the range
            [0, 4] and on the format (H, W, C). 

    Returns:
        PIL.Image.Image: A PIL image.
    """
    new_hdr = np.clip(hdr_array, 0, 1)
    new_hdr = new_hdr**(1/2.2)
    new_hdr_img = PIL.Image.fromarray((new_hdr * 255).astype(np.uint8))
    return new_hdr_img

def hdr_np_to_img_tonemap(hdr_array: np.ndarray) -> PIL.Image.Image:
    """Converts a numpy array representing an HDR image to a PIL image.

    Args:
        hdr_array (np.ndarray): Numpy array representing an HDR image. Values are in the range
            [0, 4] and on the format (H, W, C).

    Returns:
        PIL.Image.Image: A PIL image.
    """
    tonemap = cv.createTonemapDrago(2.2)
    scale = 1 / tonemap.getSaturation()
    hdr_array = scale * tonemap.process(hdr_array)
    hdr_array = np.clip(hdr_array, 0, 1)
    hdr_img = PIL.Image.fromarray((hdr_array * 255).astype(np.uint8))
    return hdr_img

def np_to_img_naive(hdr_array: np.ndarray) -> PIL.Image.Image:
    """Converts a numpy array representing an HDR image to a PIL image.

    Args:
        hdr_array (np.ndarray): Numpy array representing an HDR image. Values are in the range
            [0, 1] and on the format (H, W, C).

    Returns:
        PIL.Image.Image: A PIL image.
    """
    hdr_img = PIL.Image.fromarray((hdr_array * 255).astype(np.uint8))
    return hdr_img

def preprocess_tensor_to_array(tensor: torch.Tensor, RGBE = False) -> np.ndarray:
    """Convert a tensor on the format CxHxW to a numpy array on the format HxWxC.
    Also change the order of channels from BGR(E) to RGB(E).

    Args:
        tensor (torch.Tensor): the tensor to convert.
        RGBE (bool, optional): If the tensor is on the format of RGBE. Defaults to False.

    Returns:
        np.ndarray: the converted tensor.
    """
    tensor = tensor.cpu()
    if RGBE:
        tensor = tensor[[2, 1, 0, 3], :, :] # BGRE to RGBE
    else:
        tensor = tensor[[2, 1, 0], :, :] # BGR to RGB
    _array = tensor.permute(1, 2, 0).numpy() # change to channel last representation
    return _array

### Both functions requires channels last format
def RGB_to_RGBE(image: np.ndarray) -> np.ndarray:
    """Converts an RGB image to RGBE format.

    Args:
        image (np.ndarray): RGB image with values in [0, 4] and shape (W, H, 3).

    Returns:
        np.ndarray: RGBE image with values in [0, 1] and shape (W, H, 4).
    """
    max_float = np.max(image, axis=-1)
    scale, exponent = np.frexp(max_float)
    scale *= 256.0/max_float
    image_rgbe = np.empty((*image.shape[:-1], 4))
    image_rgbe[..., :3] = image * scale[..., np.newaxis]
    image_rgbe[..., -1] = exponent + 128
    image_rgbe[scale < 1e-32, :] = 0
    image_rgbe /= 255
    return image_rgbe.astype(np.float32)

def RGBE_to_RGB(image: np.ndarray) -> np.ndarray:
    """Converts an RGBE image to RGB format.

    Args:
        image (np.ndarray): RGBE image with values in [0, 1] and shape (W, H, 4).

    Returns:
        np.ndarray: RGB image with values in [0, 4] and shape (W, H, 3).
    """
    image *= 255
    exponent = image[..., -1] - 128
    scale = np.power(2, exponent)
    image_rgb = np.empty((*image.shape[:-1], 3))
    image_rgb = image[..., :3] * scale[..., np.newaxis]
    image_rgb /= 256
    return image_rgb.astype(np.float32)