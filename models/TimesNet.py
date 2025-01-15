import torch
from torch import nn
from torch import fft
from torch.nn import functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # (Batch, Time, Channel)
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_k_list = torch.topk(frequency_list, k)
    top_k_list = top_k_list.detach().cpu().numpy()
    period = x.shape[1] // top_k_list

    return period, abs(xf).mean(-1)[:, top_k_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # parameter effiecient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model,
                               configs.d_ff,
                               num_kernels=configs.num_kernls),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff,
                               configs.d_model,
                               num_kernels=configs.num_kernels)
        )
    
    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        result = list()
        for i in range(self.k):
            period = period_list[i]

            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)

            else:
                length = self.seq_len + self.pred_len
                out = x
            
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D conv: from 1d variation to 2D Variation
            out = self.conv(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            result.append(out[:, :(self.seq_len + self.pred_len), :])
        
        result = torch.stack(result, dim=-1)

        # adaptive regression
        period_weight = F.softmax(period_weight, dim=-1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        result = torch.sum(result * period_weight, -1)

        # residual connection
        result = result + x
        return result
