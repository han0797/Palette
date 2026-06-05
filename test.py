# -*- coding: utf-8 -*-
import sys
import os
import argparse
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import inspect
import random

import core.praser as Praser
from data import define_dataloader
from models import create_model, define_network, define_metric, define_loss
from models.metric import *
from data.minmax_dataset import Swath_MinMax

# =========================================================================
# 기본 설정
# =========================================================================
v_num = 2
path_num = None
cp_num = 338
gpu = 1

if v_num != -1:
    saved_code_path = f'experiments/train_AMI2GMI_v{v_num}/code' if type(v_num) != str else f'experiments/train_AMI2GMI_{v_num}/code'
    if os.path.exists(saved_code_path):
        print(f">>> [System] Inserting path to sys.path: {saved_code_path}")
        sys.path.insert(0, saved_code_path)
    else:
        print(f">>> [Warning] Path not found: {saved_code_path}. Using local modules.")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> [System] Seed is fixed to {seed}")

def parse_list(s):
    return s.split(',')

# =========================================================================
# 1. 인자 파싱 및 설정
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/AMI2GMI.json', help='JSON file for configuration')
parser.add_argument('-ckpt', '--checkpoint', type=str, default=f'experiments/train_AMI2GMI_v{v_num}/checkpoint/{cp_num}_Network.pth', help='Path to the .state or _Network.pth file')
parser.add_argument('-p', '--phase', type=str, default='test', help='Run train or test')
parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=f'{gpu}', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('-ch', '--channels', type=parse_list, default=['IR105-IR096', 'IR105-WV063', 'WV073-WV063', 'IR133-IR112', 'WV073-IR096', 'WV073-IR133', 'IR105-IR112'], help='Input bands')
parser.add_argument('-s', '--sampler', type=str, default='ddim', help='Sampling method: ddpm or ddim')
parser.add_argument('-step', '--infer_step', type=int, default=1000, help='Steps for fast sampling (only for ds)')
parser.add_argument('-obj', '--objective', type=str, default='pred_x0', choices=['pred_noise', 'pred_x0'])
args = parser.parse_args()

opt = Praser.parse(args)
opt['channels'] = args.channels
opt['phase'] = 'test'
opt['distributed'] = False 
opt['gpu_ids'] = [int(id) for id in args.gpu_ids.split(',')]
opt['path']['resume_state'] = args.checkpoint
opt['objective'] = args.objective
gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt['seed'] is not None and opt['seed'] >= 0:
    seed_everything(opt['seed'])
else:
    print(">>> [System] Random seed is used.")

# ★ 변경: 사용할 평가지표 명시 (Config와 일치해야 함)
info_df = pd.DataFrame(columns=['CC', 'RMSE', 'MAE', 'SSIM', 'POD', 'FAR', 'CSI'])

# =========================================================================
# 2. 모델 초기화 및 가중치 로드
# =========================================================================
if 'beta_schedule' in opt['model']['which_networks'][0]['args']:
    train_schedule = opt['model']['which_networks'][0]['args']['beta_schedule']['train']
    opt['model']['which_networks'][0]['args']['beta_schedule']['test'] = train_schedule
    print(f"Test schedule overwritten with Train schedule (n_timestep={train_schedule['n_timestep']})")

logger = logging.getLogger()
phase_loader, val_loader = define_dataloader(logger, opt)

networks = [define_network(logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
metrics = [define_metric(logger, item_opt) for item_opt in opt['model']['which_metrics']]
losses = [define_loss(logger, item_opt) for item_opt in opt['model']['which_losses']]

model = create_model(opt=opt, networks=networks, phase_loader=None, val_loader=None, 
                     losses=losses, metrics=metrics, logger=logger, writer=None)

ckpt_path = args.checkpoint.replace('.state', '_Network.pth') if args.checkpoint.endswith('.state') else args.checkpoint
state_dict = torch.load(ckpt_path, map_location='cpu')
if isinstance(state_dict, dict) and 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.netG.load_state_dict(state_dict, strict=False)
model.netG.eval()
model.netG.to(device)

# =========================================================================
# 3. 데이터 로드 및 추론
# =========================================================================
dataset_opt = opt['datasets']['test']['which_dataset']['args']
dataset_opt['app'] = 'test'
test_dataset = Swath_MinMax(**dataset_opt)

save_path = f'/data-8t/khoon/rsrch_GMI5/Palette/experiments/train_AMI2GMI_v{v_num}/val_test/' if path_num==None else f'val_test/version_{path_num}'
os.makedirs(f'{save_path}/data/y', exist_ok=True)
os.makedirs(f'{save_path}/data/y_hat', exist_ok=True)
os.makedirs(f'{save_path}/figure', exist_ok=True)

#for i in reversed(range(0, len(test_dataset))):
for i in reversed(range(21, 22)):
    data_item = test_dataset[i]
    target_filename = data_item['path']
    
    # 파일명 (예: tbh_TCID_202601011200_intpol.npy)에서 날짜 파싱
    try:
        date_str = target_filename.split('_')[2]
        formatted = datetime.strptime(date_str, '%Y%m%d%H%M').strftime("%b. %d, %Y at %H:%M UTC")
    except:
        formatted = target_filename.replace('.npy', '')
    
    cond_image = data_item['cond_image'].unsqueeze(0).to(device)
    gt_image = data_item['gt_image'].unsqueeze(0).to(device)
    mask = data_item['mask'].unsqueeze(0).to(device) # ★ Mask 데이터 추출
    
    sample_num = 8
    has_sampler_arg = 'sampler' in inspect.signature(model.netG.restoration).parameters
    
    with torch.no_grad():
        if has_sampler_arg:
            output, visuals = model.netG.restoration(cond_image, sample_num=sample_num, sampler=args.sampler, infer_step=args.infer_step) 
        else:
            output, visuals = model.netG.restoration(cond_image, sample_num=sample_num)
    
    # ★ 변경: [-1, 1] 출력을 실제 tbh (K) 단위로 역정규화
    pr_tbh = Nor2tbh(output)
    gt_tbh = Nor2tbh(gt_image)
    
    # ★ 변경: 마스크를 통과시켜 지표 계산
    metric_results = [
        cc(gt_tbh, pr_tbh, mask), 
        rmse(gt_tbh, pr_tbh, mask), 
        mae(gt_tbh, pr_tbh, mask), 
        ssim(gt_tbh, pr_tbh, mask),
        pod(gt_tbh, pr_tbh, mask), 
        far(gt_tbh, pr_tbh, mask), 
        csi(gt_tbh, pr_tbh, mask)
    ]
    info_df.loc[target_filename] = [x.item() for x in metric_results]
    pr_np = np.where(mask.cpu().numpy().squeeze() == 1, pr_tbh.cpu().numpy().squeeze(), np.nan)
    gt_np = np.where(mask.cpu().numpy().squeeze() == 1, gt_tbh.cpu().numpy().squeeze(), np.nan)
 
    # Tensor -> Numpy 변환 및 결측치 영역을 NaN으로 치환하여 저장
    pr_tbh_np = pr_tbh.cpu().numpy().squeeze()
    gt_tbh_np = gt_tbh.cpu().numpy().squeeze()
    mask_np = mask.cpu().numpy().squeeze()
    
    pr_tbh_np = np.where(mask_np == 1, pr_tbh_np, np.nan)
    gt_tbh_np = np.where(mask_np == 1, gt_tbh_np, np.nan)
    
    np.save(f'{save_path}/data/y_hat/{target_filename}', pr_tbh_np)
    np.save(f'{save_path}/data/y/{target_filename}', gt_tbh_np)
    
    # ---------------------------------------------------------------------
    # 4. 시각화 (jet_r, 200~300K)
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(12,6))
    
    # 정답 시각화
    ax1 = plt.subplot(1,2,1)
    im1 = ax1.imshow(gt_np, cmap='jet_r', vmin=200, vmax=300)
    ax1.set_title('GMI (Ground Truth)', fontsize=18, weight='bold')
    ax1.axis('off')
    cb1 = plt.colorbar(im1, ax=ax1, orientation='vertical', pad=0.04, fraction=0.046)
    cb1.ax.tick_params(labelsize=12)
    cb1.set_label(label='Brightness Temperature (K)', size=14)
    
    # 예측 시각화
    ax2 = plt.subplot(1,2,2)
    im2 = ax2.imshow(pr_np, cmap='jet_r', vmin=200, vmax=300)
    ax2.set_title('Palette (Predicted)', fontsize=18, weight='bold')
    ax2.axis('off')
    cb2 = plt.colorbar(im2, ax=ax2, orientation='vertical', pad=0.04, fraction=0.046)
    cb2.ax.tick_params(labelsize=12)
    cb2.set_label(label='Brightness Temperature (K)', size=14)
    
    plt.suptitle(formatted, x=0.5, y=0.98, ha='center', fontsize=20, weight='bold')
    plt.savefig(f'{save_path}/figure/{target_filename}.png', bbox_inches='tight', dpi=150)
    
    plt.clf()
    plt.close()
    
info_df.to_csv(f'{save_path}/info.csv')
