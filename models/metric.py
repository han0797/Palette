import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

# ---------------------------------------------------------
# 1. 역정규화 함수: [-1, 1] -> [50, 310] K (tbh)
# ---------------------------------------------------------
def Nor2tbh(data):
    """ Palette의 [-1, 1] 출력을 실제 tbh [50, 310] K로 변환합니다. """
    data = torch.clamp(data, 0.0, 1.0)
    tbh = data * (310.0 - 50.0) + 50.0
    return tbh

# ---------------------------------------------------------
# 공통 헬퍼 함수: Mask가 있을 경우 유효한 픽셀만 1D로 추출
# ---------------------------------------------------------
def _apply_mask(gt, pr, mask):
    if mask is not None:
        valid_idx = (mask == 1)
        return gt[valid_idx], pr[valid_idx]
    return gt.flatten(), pr.flatten()

# ---------------------------------------------------------
# 2. 연속형 평가지표 (MAE, RMSE, CC)
# ---------------------------------------------------------
def mae(gt, pr, mask=None):
    gt_v, pr_v = _apply_mask(gt, pr, mask)
    if gt_v.numel() == 0: return torch.tensor(0.0, device=gt.device)
    return torch.mean(torch.abs(pr_v - gt_v))

def rmse(gt, pr, mask=None):
    gt_v, pr_v = _apply_mask(gt, pr, mask)
    if gt_v.numel() == 0: return torch.tensor(0.0, device=gt.device)
    return torch.sqrt(torch.mean((pr_v - gt_v) ** 2))

def cc(gt, pr, mask=None):
    gt_v, pr_v = _apply_mask(gt, pr, mask)
    if gt_v.numel() == 0: return torch.tensor(0.0, device=gt.device)
    
    vx = pr_v - torch.mean(pr_v)
    vy = gt_v - torch.mean(gt_v)
    
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return cost

# ---------------------------------------------------------
# 3. 범주형 평가지표 (POD, FAR, CSI) - 임계값 < 235K
# ---------------------------------------------------------
def _get_contingency_table(gt, pr, mask=None, threshold=235.0):
    gt_v, pr_v = _apply_mask(gt, pr, mask)
    if gt_v.numel() == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    # tbh는 온도가 낮을수록( < 235K ) 대류운/강수를 의미하므로 미만 조건을 사용
    pr_bool = pr_v < threshold
    gt_bool = gt_v < threshold
    
    hits = torch.sum(pr_bool & gt_bool).float()
    misses = torch.sum((~pr_bool) & gt_bool).float()
    false_alarms = torch.sum(pr_bool & (~gt_bool)).float()
    correct_nulls = torch.sum((~pr_bool) & (~gt_bool)).float()
    
    return hits, misses, false_alarms, correct_nulls

def pod(gt, pr, mask=None):
    """ Probability of Detection (Hit Rate) """
    h, m, f, c = _get_contingency_table(gt, pr, mask, threshold=235.0)
    return h / (h + m + 1e-8)

def far(gt, pr, mask=None):
    """ False Alarm Ratio """
    h, m, f, c = _get_contingency_table(gt, pr, mask, threshold=235.0)
    return f / (h + f + 1e-8)

def csi(gt, pr, mask=None):
    """ Critical Success Index (Threat Score) """
    h, m, f, c = _get_contingency_table(gt, pr, mask, threshold=235.0)
    return h / (h + m + f + 1e-8)

# ---------------------------------------------------------
# 4. SSIM (Structural Similarity Index Measure)
# ---------------------------------------------------------
def _gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(gt, pr, mask=None):
    """ SSIM은 공간적 구조를 봐야하므로 2D 형태를 유지한 채로 계산 후 Mask 적용 """
    if pr.dim() == 2:
        pr = pr.unsqueeze(0).unsqueeze(0)
        gt = gt.unsqueeze(0).unsqueeze(0)
        if mask is not None: mask = mask.unsqueeze(0).unsqueeze(0)
    elif pr.dim() == 3:
        pr = pr.unsqueeze(1)
        gt = gt.unsqueeze(1)
        if mask is not None: mask = mask.unsqueeze(1)

    (_, channel, _, _) = pr.size()
    window_size = 11
    window = _create_window(window_size, channel).to(pr.device)

    mu1 = F.conv2d(gt, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(pr, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(gt*gt, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(pr*pr, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(gt*pr, window, padding=window_size//2, groups=channel) - mu1_mu2

    # tbh 데이터 범위 (310 - 50 = 260)에 맞춘 L값 설정
    L = 260.0 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Mask가 있을 경우 결측치가 있는 부분을 제외하고 평균 SSIM 도출
    if mask is not None:
        ssim_map = ssim_map * mask
        return ssim_map.sum() / (mask.sum() + 1e-8)
    else:
        return ssim_map.mean()
