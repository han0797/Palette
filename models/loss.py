import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# =============================================================================
# 기본 손실 (원본 유지)
# =============================================================================

def mse_loss(output, target):
    return F.mse_loss(output, target)


def l1_loss(output, target):
    return F.l1_loss(output, target)


def smooth_l1_loss(output, target):
    return F.smooth_l1_loss(output, target, beta=1.0)


class WeightedMSELoss(nn.Module):
    def __init__(self, rain_weight=5.0, threshold=-0.7575):
        super().__init__()
        self.rain_weight = rain_weight
        self.threshold = (np.log10(0.1) * 16 + 10 * np.log10(200)) / 57.828 * 2 - 1
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask=None, gt_image=None):
        loss = self.mse(pred, target)
        if gt_image is not None:
            rain_mask = (gt_image > self.threshold).float()
            weights = 1.0 + (self.rain_weight * rain_mask)
            loss = loss * weights
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


# =============================================================================
# [수정] ls_weighted_l1_loss
#
#  핵심 버그 수정:
#   - 기존 코드는 가중 맵을 `target`에서 계산해서 objective='pred_noise'일 때
#     노이즈 값 기준으로 가중치가 잘못 부여됨.
#   - gt_image 인자를 추가하고 가중 맵을 항상 '이미지(y_0)' 기준으로 계산.
#   - pred_x0에서는 target=y_0이므로 gt_image 없어도 동일하게 동작.
# =============================================================================

class ls_weighted_l1_loss(nn.Module):
    def __init__(self, alpha=0.5, threshold_norm=0.53846, weight_low=3.0, weight_high=1.0):
        super().__init__()
        self.alpha = alpha
        self.threshold_norm = threshold_norm
        self.weight_low = weight_low
        self.weight_high = weight_high

    def _weight_map(self, ref):
        return torch.where(
            ref <= self.threshold_norm,
            torch.full_like(ref, self.weight_low),
            torch.full_like(ref, self.weight_high),
        )

    def forward(self, pred, target, mask=None, gt_image=None):
        # 가중 맵은 항상 '이미지'에서 — pred_noise여도 gt_image(y_0) 기준으로 올바르게 동작
        ref = gt_image if gt_image is not None else target
        weight = self._weight_map(ref)

        mse = F.mse_loss(pred, target, reduction='none')
        l1 = F.l1_loss(pred, target, reduction='none')

        weighted = self.alpha * (mse * weight) + (1.0 - self.alpha) * (l1 * weight)

        if mask is not None:
            weighted = weighted * mask
            return weighted.sum() / (mask.sum() + 1e-8)
        return weighted.mean()


# =============================================================================
# [신규] structural_weighted_loss  (objective='pred_x0' 권장)
#
#  목적: 태풍 눈/눈벽처럼 작고 날카로운 경계 구조를 살리기 위한 이미지-공간 손실.
#    total = alpha*MSE + (1-alpha)*L1            (한기역 가중)
#          + lambda_grad * GradientDifferenceLoss (눈벽 등 날카로운 경계 보존)
#          + lambda_ssim * (1 - SSIM)             (국소 구조/대비; 기본 off)
#
#  - pred, target 모두 [-1,1] 정규화된 밝기온도 이미지여야 함 (pred_x0 전용).
#  - 모든 항에 swath mask를 적용해 결측 픽셀을 평균에서 제외.
#  - lambda_ssim=0.0이 기본값: SSIM은 swath 경계에서 왜곡 우려가 있으므로
#    lambda_grad 검증 후 점진적으로 추가할 것.
# =============================================================================

class structural_weighted_loss(nn.Module):
    def __init__(self, alpha=0.3, threshold_norm=0.53846, weight_low=3.0, weight_high=1.0,
                 lambda_grad=0.5, lambda_ssim=0.0, ssim_window=11):
        super().__init__()
        self.alpha = alpha
        self.threshold_norm = threshold_norm
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.lambda_grad = lambda_grad
        self.lambda_ssim = lambda_ssim
        self.ssim_window = ssim_window

    def _weight_map(self, ref):
        return torch.where(
            ref <= self.threshold_norm,
            torch.full_like(ref, self.weight_low),
            torch.full_like(ref, self.weight_high),
        )

    @staticmethod
    def _masked_mean(x, m):
        if m is None:
            return x.mean()
        return (x * m).sum() / (m.sum() + 1e-8)

    def _pixel_loss(self, pred, target, weight, mask):
        mse = F.mse_loss(pred, target, reduction='none')
        l1 = F.l1_loss(pred, target, reduction='none')
        px = self.alpha * (mse * weight) + (1.0 - self.alpha) * (l1 * weight)
        return self._masked_mean(px, mask)

    def _grad_loss(self, pred, target, weight, mask):
        # x 방향 차분
        pdx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        tdx = target[:, :, :, 1:] - target[:, :, :, :-1]
        wx = weight[:, :, :, 1:]
        gx = (pdx - tdx).abs() * wx
        mx = None if mask is None else mask[:, :, :, 1:] * mask[:, :, :, :-1]

        # y 방향 차분
        pdy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tdy = target[:, :, 1:, :] - target[:, :, :-1, :]
        wy = weight[:, :, 1:, :]
        gy = (pdy - tdy).abs() * wy
        my = None if mask is None else mask[:, :, 1:, :] * mask[:, :, :-1, :]

        return self._masked_mean(gx, mx) + self._masked_mean(gy, my)

    def _gaussian_window(self, channels, device, dtype):
        ws = self.ssim_window
        sigma = 1.5
        coords = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        win2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
        return win2d.expand(channels, 1, ws, ws).contiguous()

    def _ssim(self, pred, target, mask):
        c = pred.shape[1]
        win = self._gaussian_window(c, pred.device, pred.dtype)
        pad = self.ssim_window // 2
        mu1 = F.conv2d(pred, win, padding=pad, groups=c)
        mu2 = F.conv2d(target, win, padding=pad, groups=c)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
        s2 = F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
        s12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu12
        # 데이터 범위 [-1, 1] → L=2
        c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
        ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
        return self._masked_mean(ssim_map, mask)

    def forward(self, pred, target, mask=None, gt_image=None):
        ref = gt_image if gt_image is not None else target
        weight = self._weight_map(ref)

        total = self._pixel_loss(pred, target, weight, mask)

        if self.lambda_grad > 0:
            total = total + self.lambda_grad * self._grad_loss(pred, target, weight, mask)

        if self.lambda_ssim > 0:
            total = total + self.lambda_ssim * (1.0 - self._ssim(pred, target, mask))

        return total
