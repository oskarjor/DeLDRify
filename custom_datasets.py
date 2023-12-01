from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np

# requires channels last
def RGB_to_RGBE(image: np.ndarray):
    max_float = np.max(image, axis=-1)
    scale, exponent = np.frexp(max_float)
    scale *= 256.0/max_float
    image_rgbe = np.empty((*image.shape[:-1], 4))
    image_rgbe[..., :3] = image * scale[..., np.newaxis]
    image_rgbe[..., -1] = exponent + 128
    image_rgbe[scale < 1e-32, :] = 0
    image_rgbe /= 255
    return image_rgbe


def RGBE_to_RGB(image: np.ndarray):
    image *= 255
    exponent = image[..., -1] - 128
    scale = np.power(2, exponent)
    image_rgb = np.empty((*image.shape[:-1], 3))
    image_rgb = image[..., :3] * scale[..., np.newaxis]
    image_rgb /= 256
    return image_rgb.astype(np.float32)


class PairWiseImagesRGBE(Dataset):

    def __init__(self, ldr_path, hdr_path, transform=None) -> None:
        self.ldr_path = ldr_path
        self.hdr_path = hdr_path
        self.transform = transform
        self.ldr_list = sorted(os.listdir(ldr_path))
        self.hdr_list = sorted(os.listdir(hdr_path))

    def __len__(self):
        return len(self.ldr_list)
    
    def __getitem__(self, idx):
        ldr_img_path = os.path.join(self.ldr_path, self.ldr_list[idx])
        hdr_img_path = os.path.join(self.hdr_path, self.hdr_list[idx])
        ldr_img = cv.imread(ldr_img_path)
        ldr_img = ldr_img.astype(np.float32)
        ldr_img /= 255.0
        hdr_img = cv.imread(hdr_img_path, flags=cv.IMREAD_ANYDEPTH)
        hdr_img = RGB_to_RGBE(hdr_img)
        if self.transform:
            ldr_img = self.transform(ldr_img)
            hdr_img = self.transform(hdr_img)
        return ldr_img, hdr_img

class PairWiseImages(Dataset):

    def __init__(self, ldr_path, hdr_path, transform=None) -> None:
        self.ldr_path = ldr_path
        self.hdr_path = hdr_path
        self.transform = transform
        self.ldr_list = sorted(os.listdir(ldr_path))
        self.hdr_list = sorted(os.listdir(hdr_path))

    def __len__(self):
        return len(self.ldr_list)
    
    def __getitem__(self, idx):
        ldr_img_path = os.path.join(self.ldr_path, self.ldr_list[idx])
        hdr_img_path = os.path.join(self.hdr_path, self.hdr_list[idx])
        ldr_img = cv.imread(ldr_img_path)
        ldr_img = ldr_img.astype(np.float32)
        ldr_img /= 255.0
        hdr_img = cv.imread(hdr_img_path, flags=cv.IMREAD_ANYDEPTH)
        if self.transform:
            ldr_img = self.transform(ldr_img)
            hdr_img = self.transform(hdr_img)
        return ldr_img, hdr_img