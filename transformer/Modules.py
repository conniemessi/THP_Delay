import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, new_mask, mask=None):
        # attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # print("q", q.size())
        # print("k", k.size())
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # np.savetxt('log_64_qk.txt', attn[0][0].detach().numpy(), fmt='%.4f')
        # attn = torch.matmul(new_mask, torch.matmul(q / self.temperature, k.transpose(2, 3)))
        # print("attn new", attn.size())

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = F.softmax(attn, dim=-1)
        # np.savetxt('log_64_softmax.txt', attn[0][0].detach().numpy(), fmt='%.4f')
        attn = torch.mul(attn, new_mask)
        # np.savetxt('log_64_softmax_mask.txt', attn[0][0].detach().numpy(), fmt='%.4f')
        # attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        # np.savetxt('log_64_v.txt', attn[0][0].detach().numpy(), fmt='%.4f')

        return output, attn
