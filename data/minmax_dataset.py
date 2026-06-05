# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from data.minmax_extractor import get_channel_minmax

class Swath_MinMax(Dataset):
    def __init__(self, image_dir, app, channels=None, augment=False, **kwargs):
        super(Swath_MinMax, self).__init__()
        self.image_dir = image_dir
        self.app = app
        
        self.data_info = pd.read_csv(os.path.join(image_dir, app + '_case.csv'))
        
        self.path_input = os.path.join(image_dir, '0_grid', 'x')
        self.path_target = os.path.join(image_dir, '0_grid', 'y', 'tbh')
        self.minmax_txt_dir = os.path.join(image_dir, 'minmax')
        
        self.channels = channels if channels is not None else ['IR105-IR096', 'IR105-IR087', 'IR133-IR096', 'WV073-IR096','IR087-IR123', 'IR087-IR096',  'IR112-IR123']
        
        self.ch_min_max = get_channel_minmax(self.minmax_txt_dir)
        self.augment = True if app == 'train' else False

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        ret = {}
        
        row = self.data_info.iloc[idx]
        date_time = str(row['datetime'])
        tcid = str(row['tcid'])
        intpol = str(row['interpolation'])
        
        target_file = f'tbh_{tcid}_{date_time}_{intpol}.npy'
        
        # 1. Target 데이터 및 Mask 생성
        target_path = os.path.join(self.path_target, target_file)
        target_data = np.load(target_path)  # Shape: (400, 400)
        
        mask_data = np.where(np.isnan(target_data), 0.0, 1.0)
        
        target_data = np.clip((target_data - 50) / (310 - 50), 0, 1)
        #target_data = target_data * 2 - 1 
        target_data = np.nan_to_num(target_data, nan=-1.0)
        
        target = np.expand_dims(target_data, axis=0) # [1, 400, 400]
        mask = np.expand_dims(mask_data, axis=0)     # [1, 400, 400]

        # 2. Input 데이터 채널별 로드
        x_list = []
        for channel in self.channels:
            mn, mx = self.ch_min_max[channel]
            
            if '-' in channel:
                b1, b2 = channel.split('-')
                d1_path = os.path.join(self.path_input, b1.lower(), f"ami_{b1.lower()}_{tcid}_{date_time}.npy")
                d2_path = os.path.join(self.path_input, b2.lower(), f"ami_{b2.lower()}_{tcid}_{date_time}.npy")
                d1 = np.load(d1_path)
                d2 = np.load(d2_path)
                
                d = np.clip(((d1 - d2) - mn) / (mx - mn), 0, 1)
            else:
                b3 = channel.lower()
                d1_path = os.path.join(self.path_input, b3, f"ami_{b3}_{tcid}_{date_time}.npy")
                d1 = np.load(d1_path)
                
                d = np.clip((d1 - mn) / (mx - mn), 0, 1)
            
            #d = d * 2 - 1
            x_list.append(d)
            
        x = np.stack(x_list, axis=0) # [C, 400, 400]

        # 3. Data Augmentation
        if self.augment:
            x, target, mask = self.apply_augmentation(x, target, mask)

        # 4. 중심 기준 Center Crop (400x400 -> 384x384)
        crop_size = 384
        h, w = x.shape[1], x.shape[2]
        
        # 중심에서 자르기 위한 시작점 계산 (400 - 384) // 2 = 8
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        
        x = x[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
        target = target[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
        mask = mask[:, start_h:start_h + crop_size, start_w:start_w + crop_size]

        # 5. 최종 반환
        ret['gt_image'] = torch.from_numpy(target).contiguous().float()
        ret['cond_image'] = torch.from_numpy(x).contiguous().float()
        ret['mask'] = torch.from_numpy(mask).contiguous().float()
        ret['path'] = str(target_file)
        
        return ret

    def apply_augmentation(self, x, y, mask):
        # 좌우 반전
        if random.random() < 0.5:
            x = np.flip(x, axis=2)
            y = np.flip(y, axis=2)
            mask = np.flip(mask, axis=2)
        return x.copy(), y.copy(), mask.copy()
    
    def gpu_shear_affine_transform(self, np_img, angle):
        device = torch.device('cpu')
        if np_img.ndim == 2:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).to(device).float()
        elif np_img.ndim == 3:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0).to(device).float()
        else:
            raise ValueError("np_img must be 2D or 3D numpy array")
        
        shear_rad = np.deg2rad(angle)
        shear_factor = np.tan(shear_rad)
        matrix = torch.tensor([
            [1, shear_factor, 0],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).to(device)
        grid = torch.nn.functional.affine_grid(matrix, tensor_img.size(), align_corners=False)
        out = torch.nn.functional.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        out = out.squeeze(0)
        if np_img.ndim == 2:
            out = out.squeeze(0)
        return out.detach().cpu().numpy()
