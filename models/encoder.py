import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder


# DINO centering

class Center(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        return x - mean


class PredictorEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super().__init__()
        self.encoder = TSEncoder(input_dims, hidden_dims)
        self.pred_layer = nn.Linear(hidden_dims, input_dims)

    def forward(self, x):
        feat = self.encoder(x)
        pred = self.pred_layer(feat)
        return pred


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', pre_project_mask=None):#, pre_project_mask=None
        super().__init__()
        self.input_dims = input_dims #1
        self.output_dims = output_dims #320
        self.hidden_dims = hidden_dims #64
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.pre_project_mask = pre_project_mask
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.center = Center()  # Center layer
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None, pre_project_mask=False):  # x: B x T x input_dims #, pre_project_mask=False
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        if pre_project_mask and mask is not None:
            x[~mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)
        x = self.center(x)
        x = self.repr_dropout(x)
        x = x.transpose(1, 2)  # B x T x Co
        
        return x


# class TeacherTSEncoder(nn.Module):
#     def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
#         super().__init__()
#         self.input_dims = input_dims  # 1
#         self.output_dims = output_dims  # 320
#         self.hidden_dims = hidden_dims  # 64
#         #self.mask_mode = mask_mode
#         self.input_fc = nn.Linear(input_dims, hidden_dims)
#         self.feature_extractor = DilatedConvEncoder(
#             hidden_dims,
#             [hidden_dims] * depth + [output_dims],
#             kernel_size=3
#         )
#         self.repr_dropout = nn.Dropout(p=0.1)
#
#     def forward(self, x):  # x: B x T x input_dims
#         nan_mask = ~x.isnan().any(axis=-1)
#         x[~nan_mask] = 0
#         x = self.input_fc(x)  # B x T x Ch
#
#         # generate & apply mask
#         # if mask is None:
#         #     if self.training:
#         #         mask = self.mask_mode
#         #     else:
#         #         mask = 'all_true'
#         #
#         # if mask == 'binomial':
#         #     mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
#         # # np.random.binomial(1, p, size=(B, T))
#         # elif mask == 'continuous':
#         #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
#         # elif mask == 'all_true':
#         #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
#         # elif mask == 'all_false':
#         #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
#         # elif mask == 'mask_last':
#         #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
#         #     mask[:, -1] = False
#
#         #mask &= nan_mask
#         #x[~mask] = 0
#
#         # conv encoder
#         x = x.transpose(1, 2)  # B x Ch x T
#         x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
#         # x = self.feature_extractor(x)
#         x = x.transpose(1, 2)  # B x T x Co
#
#         return x