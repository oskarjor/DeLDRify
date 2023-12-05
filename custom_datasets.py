from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np
from utils import RGB_to_RGBE


class PairWiseImagesRGBE(Dataset):

    def __init__(self, ldr_path, hdr_path, transform=None, device = "cpu") -> None:
        self.ldr_path = ldr_path
        self.hdr_path = hdr_path
        self.transform = transform
        self.ldr_list = sorted(os.listdir(ldr_path))
        self.hdr_list = sorted(os.listdir(hdr_path))
        self.device = device

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
        return ldr_img.to(self.device), hdr_img.to(self.device)

class PairWiseImages(Dataset):

    def __init__(self, ldr_path, hdr_path, transform=None, device = "cpu") -> None:
        self.ldr_path = ldr_path
        self.hdr_path = hdr_path
        self.transform = transform
        self.ldr_list = sorted(os.listdir(ldr_path))
        self.hdr_list = sorted(os.listdir(hdr_path))
        self.device = device

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
        return ldr_img.to(self.device), hdr_img.to(self.device)
    
class PairWiseImagesNTIRE(Dataset):

    def __init__(self, ldr_path, hdr_path, transform=None, device = "cpu") -> None:
        self.ldr_path = ldr_path
        self.hdr_path = hdr_path
        self.transform = transform
        self.ldr_list = sorted(os.listdir(ldr_path))
        self.hdr_list = sorted(os.listdir(hdr_path))
        self.device = device

    def __len__(self):
        return len(self.ldr_list)
    
    def __getitem__(self, idx):
        ldr_img_path = os.path.join(self.ldr_path, self.ldr_list[idx])
        hdr_img_path = os.path.join(self.hdr_path, self.hdr_list[idx])
        ldr_img = cv.imread(ldr_img_path)
        ldr_img = ldr_img.astype(np.float32)
        ldr_img /= 255.0
        hdr_img = cv.imread(hdr_img_path, flags=cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
        hdr_img = hdr_img.astype(np.float32)
        hdr_img /= 16384
        if self.transform:
            ldr_img = self.transform(ldr_img)
            hdr_img = self.transform(hdr_img)
        return ldr_img.to(self.device), hdr_img.to(self.device)
    
class PairWiseImagesHDREye(Dataset):

    def __init__(self, ldr_path, hdr_path, transform=None, device = "cpu") -> None:
        self.ldr_path = ldr_path
        self.hdr_path = hdr_path
        self.transform = transform
        self.ldr_list = sorted(os.listdir(ldr_path))
        self.hdr_list = sorted(os.listdir(hdr_path))
        self.device = device

    def __len__(self):
        return len(self.ldr_list)
    
    def __getitem__(self, idx):
        ldr_img_path = os.path.join(self.ldr_path, self.ldr_list[idx])
        hdr_img_path = os.path.join(self.hdr_path, self.hdr_list[idx])
        ldr_img = cv.imread(ldr_img_path + "/gt.png")
        ldr_img = ldr_img.astype(np.float32)
        ldr_img /= 255.0
        hdr_img = cv.imread(hdr_img_path + "/gt.hdr", flags=cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
        hdr_img = hdr_img.astype(np.float32)
        hdr_img /= 16384
        if self.transform:
            ldr_img = self.transform(ldr_img)
            hdr_img = self.transform(hdr_img)
        return ldr_img.to(self.device), hdr_img.to(self.device)