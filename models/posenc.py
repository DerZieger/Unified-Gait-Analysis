import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, dropout : float, max_len : int = 5096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[ : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
    def enc(self,x):
        return self.dropout(self.pe[x])