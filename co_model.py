from matplotlib.pyplot import xkcd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from ACSConv.acsconv.converters import ACSConverter


class VGG(nn.Module):
    def __init__(self, backend, co_layers=16, total_layers=29, image_shape=(1, 1)):
        super().__init__()
        conv_co = []
        conv_CG = []
        for i, layer in enumerate(backend.features):
            if i < co_layers:
                conv_co.append(layer)
            elif i < total_layers:
                conv_CG.append(layer)
        self.conv_co = nn.Sequential(*conv_co)
        self.conv_CG = nn.Sequential(*conv_CG)
        self.conv_IS = copy.deepcopy(self.conv_CG)
        self.image_shape = image_shape

    def forward_CG(self, x, size):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            output = self.conv_co(x[:, i])
            output = self.conv_CG(output)
            output = nn.functional.adaptive_avg_pool2d(output, self.image_shape)
            output = output.reshape(-1, 512, self.image_shape[0], self.image_shape[1])
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward_IS(self, x, size):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            output = self.conv_co(x[:, i])
            output = self.conv_IS(output)
            output = nn.functional.adaptive_avg_pool2d(output, self.image_shape)
            output = output.reshape(-1, 512, self.image_shape[0], self.image_shape[1])
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


# class Classifier(nn.Module):
#     def __init__(self, num_classes, hidden_dim=16):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(512, 512, 2, 1, 0),
#             # nn.GroupNorm(32, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Conv2d(512, 512, 3, 1, 0),
#             # nn.GroupNorm(32, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#         )
#         self.encoder = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(128, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#         )
#         self.classifier = nn.Linear(hidden_dim + 2, num_classes)
#     def forward(self, outputs, age, gender, channel):
#         outputs = outputs[:, :max(channel)].contiguous()
#         B, N, C, H, W  = outputs.shape
#         mask = torch.arange(N, device=channel.device).view(1, -1) < channel.view(-1, 1)
#         mask = mask.unsqueeze(dim=-1)
#         outputs = outputs.reshape(B, N, -1)
#         outputs = torch.masked_fill(outputs, ~mask, value=0)
#         f_mean = outputs = torch.sum(outputs, dim=1) / channel.view(-1, 1)
#         outputs = outputs.reshape(B, C, H, W)
#         outputs = self.conv(outputs).view(-1, C)
#         f_encode = outputs = self.encoder(outputs)
#         outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
#         outputs = self.classifier(outputs)
#         return outputs, (f_mean, f_encode)


class VolumeModel(nn.Module):
    def __init__(self, backend, device):
        super().__init__()
        self.backend = ACSConverter(backend).to(device)

    def forward_CG(self, x, size):
        x = torch.transpose(x, 1, 2)                  # (B, C, N, H, W)
        outputs = self.backend(x).view(-1, 512)
        return outputs

    def forward_IS(self, x, size):
        x = torch.transpose(x, 1, 2)                  # (B, C, N, H, W)
        outputs = self.backend(x).view(-1, 512)
        return outputs


# class Classifier(nn.Module):
#     def __init__(self, num_classes, hidden_dim=16):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(128, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#         )
#         self.classifier = nn.Linear(hidden_dim + 2, num_classes)

#     def forward(self, outputs, age, gender, channel):
#         f_conv = outputs
#         f_encode = outputs = self.encoder(outputs)
#         outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
#         outputs = self.classifier(outputs)
#         return outputs, (f_conv, f_encode)


class VGGwithEmbedding(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.embedding = nn.Embedding(32, 1 * 1)
        # self.conv_CG = nn.Sequential(
        #     nn.Conv2d(512, 512, 2, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Conv2d(512, 512, 3, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        # )
        # self.conv_IS = nn.Sequential(
        #     nn.Conv2d(512, 512, 2, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Conv2d(512, 512, 3, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        # )
    def forward_CG(self, x, size):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            loc = self.embedding(i.to(x.device)).reshape(1, 1, 1)
            output = self.backend(x[:, i])
            output = output.reshape(-1, 512, 1, 1)
            outputs.append(output + loc)
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.mean(outputs, dim=1).view(-1, 512)
        # outputs = self.conv_CG(outputs).view(-1, 512)
        return outputs

    def forward_IS(self, x, size):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            loc = self.embedding(i.to(x.device)).reshape(1, 1, 1)
            output = self.backend(x[:, i])
            output = output.reshape(-1, 512, 1, 1)
            outputs.append(output + loc)
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.mean(outputs, dim=1).view(-1, 512)
        # outputs = self.conv_IS(outputs).view(-1, 512)
        return outputs


class VGGwithAttn(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.attn = Attention(512)
        # self.conv_CG = nn.Sequential(
        #     nn.Conv2d(512, 512, 2, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Conv2d(512, 512, 3, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        # )
        # self.conv_IS = nn.Sequential(
        #     nn.Conv2d(512, 512, 2, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Conv2d(512, 512, 3, 1, 0),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        # )
    def forward_CG(self, x, size, return_attn=False):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            output = self.backend(x[:, i])
            output = output.view(-1, 512)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs, w = self.attn(outputs)
        outputs = outputs.view(-1, 512)
        # outputs = self.conv_CG(outputs).view(-1, 512)
        if return_attn:
            return outputs, w
        else:
            return outputs
        
    def forward_IS(self, x, size, return_attn=False):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            output = self.backend(x[:, i])
            output = output.view(-1, 512)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs, w = self.attn(outputs)
        outputs = outputs.view(-1, 512)
        # outputs = self.conv_IS(outputs).view(-1, 512)
        if return_attn:
            return outputs, w
        else:
            return outputs
        

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Conv2d(512, 1, 4)
    
    def forward(self, outputs):
        B, N, C, H, W = outputs.shape
        w = self.attn(outputs.view(N * B, C, H, W))
        w = w.view(B, N, -1)
        w = torch.softmax(w, dim=-1)
        w = w.view(B, N, 1, 1, 1)
        outputs = (outputs * w).sum(dim=1)
        return outputs


class Attention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )
    
    def forward(self, outputs, m=None):
        B, N, H = outputs.shape
        w = self.attn(outputs.view(N * B, H)).view(B, N)
        w = torch.softmax(w, dim=-1)
        w = w.view(B, N, 1)
        outputs = (outputs * w).sum(dim=1)
        return outputs, w.view(B, N)


class VGGwithMultiAttn(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward_CG(self, x, size):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            output = self.backend(x[:, i])
            outputs.append(output.view(-1, 512))
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.transpose(outputs, 0, 1)                   # (N, B, 512)
        return outputs

    def forward_IS(self, x, size):
        x = x[:, :max(size)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            output = self.backend(x[:, i])
            outputs.append(output.view(-1, 512))
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.transpose(outputs, 0, 1)                   # (N, B, 512)
        return outputs


class Classifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(512, 4)
        self.down = nn.Linear(512, hidden_dim)
        self.attn = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, outputs, age, gender, size):
        m = torch.arange(max(size), device=size.device)[None, :] < size[:, None]
        outputs, _ = self.self_attn(outputs, outputs, outputs, key_padding_mask=~m) # (N, B, 512)
        f_selfattn = outputs = torch.transpose(outputs, 0, 1)                       # (B, N, 512)
        f_down = outputs = self.down(outputs)                                       # (B, N, 128)
        f_attn = outputs = self.attn(outputs, ~m)
        
        # outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)                  # (B, num_classes)
        return outputs, (f_selfattn.mean(dim=1), f_down.mean(dim=1), f_attn)


