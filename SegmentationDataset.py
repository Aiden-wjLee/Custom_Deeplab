from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
import glob
import os
import numpy as np
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip
import torch
from torchvision.transforms import ColorJitter
import random
import cv2
from torchvision.transforms import RandomRotation
from torchvision.transforms import functional as TF
def map_mask_values(mask):
    mask = mask# * 255  # If the mask has been normalized by ToTensor(), denormalize it
    if mask.dtype == torch.float32:
        mask = mask * (len(np.unique(mask))-1)  # If the mask has been normalized by ToTensor(), denormalize it
    
    mask = mask.long()  # Convert to integer tensor
    #mask[mask == 128] = 1 #nachi   
    #mask[mask == 255] = 2 #nachi
    return mask
def apply_motion_blur(image, max_ksize=10):
    '''Applies motion blur to an image.
    The direction of motion is chosen randomly.
    '''
    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Randomly choose the direction: horizontal or vertical
    direction = random.choice(['horizontal', 'vertical'])
    ksize = random.randint(1, max_ksize)

    if direction == 'horizontal':
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1)/2), :] = np.ones(ksize)
        kernel = kernel / ksize
    else:
        kernel = np.zeros((ksize, ksize))
        kernel[:, int((ksize - 1)/2)] = np.ones(ksize)
        kernel = kernel / ksize

    blurred = cv2.filter2D(image_np, -1, kernel)
    return Image.fromarray(blurred)

def apply_gaussian_blur(image, max_ksize=5):
    '''Applies Gaussian blur to an image.'''
    # Convert PIL image to numpy array
    image_np = np.array(image)

    ksize = random.choice(list(range(1, max_ksize+1, 2)))
    blurred = cv2.GaussianBlur(image_np, (ksize, ksize), 0)

    return Image.fromarray(blurred)

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, crop_size, transform='default'):#transform=None):
        self.root_dir = root_dir
        #self.transform = transform
        self.crop_size = crop_size
        self.image_paths = glob.glob(os.path.join(root_dir, 'img', '*.jpg')) + glob.glob(os.path.join(root_dir, 'img', '*.png'))
        self.transform = transform
        #self.image_paths = glob.glob(os.path.join(root_dir, 'img',  '*.jpg')) #'*',

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = image_path.replace('img', 'mask')  #.replace('.jpg', '_mask.jpg')  #if name is 1_mask.jpg
        mask_path = mask_path.replace('jpg', 'png') 
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') #gray scale

        if self.transform =='default':
            image_transform = transforms.Compose([
                transforms.Resize((self.crop_size[0], self.crop_size[1]), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ])
            mask_transform = transforms.Compose([
                transforms.Resize((self.crop_size[0], self.crop_size[1]), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Lambda(map_mask_values),
            ])
            
        if self.transform == 'augmentation':
            # 이미지와 마스크에 대한 transform 정의
            # 50% 확률로 좌우반전을 결정
            #flip = np.random.choice([True, False], p=[0.5, 0.5])
            
            #if flip:
            #    image = ImageOps.mirror(image)
            #    mask = ImageOps.mirror(mask)
            
            # 밝기와 대비를 무작위로 조절
            direction = random.choice(['motion_blur', 'gaussian_blur'])
            color_jitter = ColorJitter(brightness=0.2, contrast=0.4)  # brightness와 contrast의 변화 범위를 설정. 이 값은 필요에 따라 조절 가능.
            image = color_jitter(image)
            if direction == 'motion_blur':
                image = apply_motion_blur(image)
            elif direction == 'gaussian_blur':
                image = apply_gaussian_blur(image)
            
            random_degree = random.uniform(-20, 20)
            image = TF.rotate(image, random_degree, resample=Image.BILINEAR)
            mask = TF.rotate(mask, random_degree, resample=Image.NEAREST)
            # 이미지와 마스크에 대한 transform 정의
            image_transform = transforms.Compose([
                transforms.Resize((self.crop_size[0], self.crop_size[1]), interpolation=Image.BILINEAR),
                #transforms.RandomRotation(degrees=random_degree, resample=False, expand=False, center=None, fill=None),
                transforms.ToTensor(),
            ])
            
            mask_transform = transforms.Compose([
                transforms.Resize((self.crop_size[0], self.crop_size[1]), interpolation=Image.NEAREST),
                #transforms.RandomRotation(degrees=random_degree, resample=Image.NEAREST, expand=False, center=None, fill=None),
                transforms.ToTensor(),
                transforms.Lambda(map_mask_values),
            ])
            
        image = image_transform(image)
        mask = mask_transform(mask)
        #print("mask_value: ",np.unique(mask))
        return image, mask

