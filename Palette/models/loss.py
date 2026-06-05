import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


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
        
    def forward(self, pred, target, gt_image=None):
        loss = self.mse(pred, target)
        
        if gt_image is not None:
            rain_mask = (gt_image > self.threshold).float()
            weights = 1.0 + (self.rain_weight * rain_mask)
            loss = loss * weights
        
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class ls_weighted_l1_loss(torch.nn.Module):
    def __init__(self, alpha=0.5, threshold_norm=0.53846, weight_low=3.0, weight_high=1.0):
        super(ls_weighted_l1_loss, self).__init__()
        # alpha: MSE와 L1의 반영 비율 (예: 0.5면 반반)
        self.alpha = alpha 
        
        # 250K 기준 [-1, 1] 스케일 변환 값 ((250-50)/260 * 2 - 1 = 0.53846)
        self.threshold_norm = threshold_norm 
        self.weight_low = weight_low
        self.weight_high = weight_high

    def forward(self, pred, target, mask=None):
        # 1. LS Loss (MSE) - 전체 뼈대 및 분포 안정화
        mse_loss = F.mse_loss(pred, target, reduction='none')

        # 2. L1 Loss - 블록 현상 제거 및 선명도 향상
        l1_loss = F.l1_loss(pred, target, reduction='none')

        # 3. 연구원님의 맞춤형 가중치 맵 생성 (250K 이하에 높은 가중치)
        weight = torch.where(target <= self.threshold_norm, self.weight_low, self.weight_high)
        
        # 각 Loss에 가중치 곱하기
        weighted_l1 = l1_loss * weight
        weighted_mse = mse_loss * weight

        # 4. 최종 Loss 결합
        total_loss = self.alpha * weighted_mse + (1.0 - self.alpha) * weighted_l1

        # 5. Swath 결측치 마스킹 처리 (mask=1인 곳만 학습)
        if mask is not None:
            total_loss = total_loss * mask
            return total_loss.sum() / (mask.sum() + 1e-8)
        else:
            return total_loss.mean()

