import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import pandas as pd
from os.path import join
import math
import joblib
import gzip
import cv2
import netCDF4
import random

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)


class AMI2HSRDataset(data.Dataset):
    def __init__(self, hsr_path, gk2a_path, app, in_chan, out_chan, channels=None, **kwargs):
        super(AMI2HSRDataset, self).__init__()
        self.hsr_path = hsr_path
        self.gk2a_path = gk2a_path
        self.app = app
        self.channels = channels
        self.extension = 'npy'
        
        if self.channels is None:
             raise ValueError("Channels information is missing in Dataset configuration.")
        self.bands = sorted(set(band for combo in channels for band in combo.split('-')))
        self.dataset = []
        for sec in range(1, 11):
            if (app != 'train') and (sec == 10):
                continue
            section_path = os.path.join(gk2a_path, app, f'section_{sec}')
            # for fname in os.listdir(os.path.join(section_path, self.bands[0])):
            #     self.dataset.append([os.path.join(section_path, band, fname) for band in self.bands])
            for fname in [fname for fname in os.listdir(os.path.join(section_path, self.bands[0])) if self.extension in fname]:
                self.dataset.append([os.path.join(section_path, band, fname) for band in self.bands])
        
        # self.dataset = self.dataset[:10]
        minmax = pd.read_csv(f'{gk2a_path}/gk2a_minmax.csv', index_col=0)
        self.min_vals = {band: minmax[f'{band}_min'].quantile(0.02) for band in channels}
        self.max_vals = {band: minmax[f'{band}_max'].quantile(0.98) for band in channels}
        random.shuffle(self.dataset)
        self.augment = True if app == 'train' else False
    
    def __getitem__(self, index):
        ret = {}
        
        date = self.dataset[index][0].split('/')[-1].split('.')[0]
        hsr_path = f'{self.hsr_path}/{date[:6]}/{date[6:8]}/RDR_CMP_HSR_KMA_{date}.bin.gz'
        with gzip.open(hsr_path, 'rb') as f:
            file_content = f.read()
            hsr = (np.frombuffer(file_content[1024:1024 + 2*2305*2881], np.short).reshape(2881, 2305)[::-1, :]) / 100
        
        hsr = cv2.resize(hsr, (576, 720), interpolation=cv2.INTER_NEAREST)[172:-46, 104:-26][25:375,100:375]
        hsr = np.nan_to_num(hsr, nan=-250.0)
        # mask = (hsr >= -250).astype(np.bool_)
        hsr = (hsr - 0) / (57.828 - 0) * 2 - 1
        hsr = np.clip(hsr, -1., 1.)
        hsr = np.expand_dims(hsr, 0)
        
        data_dict = {}
        for i in range(len(self.bands)):
            if self.extension == 'nc':
                with netCDF4.Dataset(self.dataset[index][i]) as src:
                    data_dict[self.bands[i]] = src.variables['Brightness temperature'][:].astype(np.float32)
            elif self.extension == 'npy':
                data = np.load(self.dataset[index][i])
                data_dict[self.bands[i]] = data
        
        x = []
        for i in range(len(self.channels)):
            if '-' in self.channels[i]:
                ch1, ch2 = self.channels[i].split('-')
                data = data_dict[ch1] - data_dict[ch2]
            else:
                ch = self.channels[i]
                data = data_dict[ch]
            
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=self.min_vals[f'{self.channels[i]}'])
            
            data = (data - self.min_vals[f'{self.channels[i]}']) / (self.max_vals[f'{self.channels[i]}'] - self.min_vals[f'{self.channels[i]}'])
            data = data * 2 - 1
            data = np.clip(data, -1.0, 1.0)
            x.append(data)
        
        x = np.stack(x, 0)
        if self.augment:
            x, hsr = self.apply_augmentation(x, hsr, mask=None)
        
        pad_width = ((0, 0), (0, (32-350%32)), (0, (32-275%32)))
        x = np.pad(x, pad_width, mode='constant', constant_values=-1)
        hsr = np.pad(hsr, pad_width, mode='constant', constant_values=-1)        
        
        del data_dict, data
        
        ret['gt_image'] = torch.from_numpy(hsr).float()
        ret['cond_image'] = torch.from_numpy(x).float()
        # ret['mask'] = torch.from_numpy(mask).float().unsqueeze(0)
        ret['path'] = str(date)
        
        return ret
    
    def __len__(self):
        return len(self.dataset)
    
    def apply_augmentation(self, x, rain, mask=None):
        if random.random() < 0.5:
            x = x[:,:,::-1].copy()  # 좌우 반전
            rain = rain[:,:,::-1].copy()
            # mask = mask[:,::-1].copy()
        
        if random.random() < 0.5:
            x = x[:,::-1,:].copy()  # 상하 반전
            rain = rain[:,::-1,:].copy()
            # mask = mask[::-1,:].copy()
        
        if random.random() < 0.5:
            shear_angle = random.uniform(-10, 10)
            x = self.gpu_shear_affine_transform(x, shear_angle)
            rain = self.gpu_shear_affine_transform(rain, shear_angle)
            # mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # [1, H, W]
            # mask = self.gpu_shear_affine_transform(mask.numpy(), shear_angle)
            # mask = mask[0] > 0.5
        
        return x, rain
    
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
        ], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2, 3)
        grid = torch.nn.functional.affine_grid(matrix, tensor_img.size(), align_corners=False)
        out = torch.nn.functional.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        out = out.squeeze(0)  # (C, H, W) 또는 (1, H, W)
        if np_img.ndim == 2:
            out = out.squeeze(0)  # (H, W)
        return out.detach().cpu().numpy()
    

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


