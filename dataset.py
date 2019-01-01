import glob
import random
import cv2
import re
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

class UltrasoundDataset(data.Dataset):
    background_color = np.array([255, 255, 255])
    
    def __init__(self, mask_list, input_preprocessor):
        mask_notall_black = [x for x in mask_list if not self.isAllBlack(x)]
        self.y = mask_notall_black
        self.x = [grp.group(1)+grp.group(2) for grp in [re.match(r'(.*)_mask(\.tif)', x) for x in mask_notall_black]]
        self.input_preprocessor = input_preprocessor
        
    def isAllBlack(self, x):
        return np.all(self.imread(x)[:,:] == [0,0,0])
        
    def imread(self, file_name):
        return cv2.imread(file_name)[2: 418, 2:578]
    
    def inputimage(self, file_name):
        return self.input_preprocessor(self.imread(file_name))
    
    def labelread(self, file_name):
        img = self.imread(file_name)
        
        gt_bg = np.all(img == UltrasoundDataset.background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        
        class1 = np.zeros(gt_bg.shape, dtype=np.float32)
        class1[gt_bg] = 1.
        return class1.reshape(-1, 1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.inputimage(self.x[idx]), self.labelread(self.y[idx])

def create_loaders(val_percent = 20, batch_size = 10):
    input_processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    filelist = glob.glob('./data/train_orig/*_mask.tif')
    val_items = val_percent*len(filelist)//100

    random.shuffle(filelist)

    validation_list = filelist[0: val_items]
    val_dataset = UltrasoundDataset(validation_list, input_processor)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    train_list = filelist[val_items:]
    train_dataset = UltrasoundDataset(train_list, input_processor)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    return (train_loader, val_loader)
