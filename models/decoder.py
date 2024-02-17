# decoder.py
from torch import nn
import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self, input_dims, output_dims):#, pre_project_mask=None
        super().__init__()
        self.input_dims = input_dims #1
        self.output_dims = output_dims #320
        self.model = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.model(x)


#mlp pred (supervised)

