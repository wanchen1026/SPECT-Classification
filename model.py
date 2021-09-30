import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ACSConv.acsconv.converters import ACSConverter


class VGGplusLinear(nn.Module):
    def __init__(self, backend, num_classes):
        super().__init__()
        self.backend = backend
        self.encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(16 + 2, num_classes)
    def forward(self, x, age, gender, channel):
        x = x[:, :max(channel)].contiguous()
        outputs = []
        for i in torch.arange(x.shape[1]):
            output = self.backend(x[:, i])
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        f_mean = outputs = torch.sum(outputs, dim=1) / channel.view(-1, 1)
        outputs = self.encoder(outputs)
        outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)
        return outputs, f_mean


class VGGplusConv2d(nn.Module):
    def __init__(self, backend, num_classes):
        super().__init__()
        self.backend = backend
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(16 + 2, num_classes)
    def forward(self, x, age, gender, channel):
        x = x[:, :max(channel)].contiguous()
        outputs = []
        for i in torch.arange(x.shape[1]):
            output = self.backend(x[:, i])
            output = output.reshape(-1, 512, 4, 4)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.sum(outputs, dim=1) / channel.view(-1, 1)
        outputs = outputs.reshape(-1, 512, 4, 4)
        f_mean = outputs = self.conv(outputs).view(-1, 512)
        outputs = self.encoder(outputs)
        outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)
        return outputs, f_mean


class VGGwithACS(nn.Module):
    def __init__(self, backend, num_classes, device):
        super().__init__()
        self.backend = ACSConverter(backend).to(device)
        self.encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(16 + 2, num_classes)
    def forward(self, x, age, gender, channel):
        x = torch.transpose(x, 1, 2)         # (B, C, N, H, W)
        f_acs = outputs = self.backend(x).view(-1, 512)
        outputs = self.encoder(outputs)
        outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)
        return outputs, f_acs


class VolumeModel(nn.Module):
    def __init__(self, backend, num_classes):
        super().__init__()
        self.backend = backend
        self.encoder = nn.Sequential(
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(16 + 2, num_classes)

    def forward(self, x, age, gender, channel):
        x = torch.transpose(x, 1, 2)                  # (B, C, N, H, W)
        f_conv = outputs = self.backend(x).view(-1, 400)
        outputs = self.encoder(outputs)
        outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)
        return outputs, f_conv


class VGGwithEmbedding(nn.Module):
    def __init__(self, backend, num_classes):
        super().__init__()
        self.backend = backend
        self.embedding = nn.Embedding(32, 4 * 4)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(16 + 2, num_classes)
    def forward(self, x, age, gender, channel):
        x = x[:, :max(channel)].contiguous()
        outputs = []
        for i in torch.arange(x.shape[1]):
            loc = self.embedding(i.to(x.device)).reshape(1, 4, 4)
            output = self.backend(x[:, i])
            output = output.reshape(-1, 512, 4, 4)
            outputs.append(output + loc)
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.mean(outputs, dim=1).reshape(-1, 512, 4, 4)
        f_mean = outputs = self.conv(outputs).view(-1, 512)
        outputs = self.encoder(outputs)
        outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)
        return outputs, f_mean


class VGGwithAttn(nn.Module):
    def __init__(self, backend, num_classes):
        super().__init__()
        self.backend = backend
        self.attn = Attention(512)
        self.encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(16 + 2, num_classes)
    def forward(self, x, age, gender, channel, return_attn=False):
        x = x[:, :max(channel)].contiguous()
        C = x.shape[1]
        outputs = []
        for i in torch.arange(C):
            output = self.backend(x[:, i])
            output = output.view(-1, 512)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs, w = self.attn(outputs)
        f_mean = outputs.view(-1, 512)
        outputs = self.encoder(f_mean)
        outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)
        if return_attn:
            return outputs, f_mean, w
        else:
            return outputs, f_mean


class Attention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )
    
    def forward(self, outputs):
        B, N, H = outputs.shape
        w = self.attn(outputs.view(N * B, H)).view(B, N)
        w = torch.softmax(w, dim=-1)
        w = w.view(B, N, 1)
        outputs = (outputs * w).sum(dim=1)
        return outputs, w.view(B, N)


class VGGwithMultiAttn(nn.Module):
    def __init__(self, backend, num_classes, hidden_dim=128):
        super().__init__()
        self.backend = backend
        self.self_attn = nn.MultiheadAttention(512, 4)
        self.down = nn.Linear(512, hidden_dim)
        self.attn = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim + 2, num_classes)

    def forward(self, x, age, gender, channel):
        x = x[:, :max(channel)].contiguous()
        outputs = []
        for i in torch.arange(x.shape[1]):
            output = self.backend(x[:, i])
            outputs.append(output.view(-1, 512))
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.transpose(outputs, 0, 1)                   # (N, B, 512)
        outputs, _ = self.self_attn(outputs, outputs, outputs)     # (N, B, 512)
        outputs = torch.transpose(outputs, 0, 1)                   # (B, N, 512)
        outputs = self.down(outputs)                               # (B, N, 128)
        f_mean = outputs = self.attn(outputs)
        outputs = torch.cat([outputs, age.view(-1, 1), gender.view(-1, 1)], dim=1)
        outputs = self.classifier(outputs)                         # (B, num_classes)
        return outputs, f_mean


class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, maxtime=32):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        position = torch.arange(0, maxtime, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(maxtime, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, xs, lens):
        for i in range(len(lens)):
            xs[:lens[i], i] += self.pe[:lens[i]]
        return self.dropout(xs)


class MIPModel(nn.Module):
    def __init__(
        self, image_model, feature_dim, projection_dim, num_classes, num_heads,
        feedforward_dim, drop_transformer, drop_cpe, pooling, image_shape=(1, 1)):
        super().__init__()

        self.image_shape = image_shape
        self.pooling = pooling
        self.image_model = image_model
        self.group_norm = nn.GroupNorm(32, feature_dim)
        self.projection = nn.Linear(feature_dim, projection_dim)

        transformer_dim = projection_dim * image_shape[0] * image_shape[1]
        self.pos_encoding = ContinuousPosEncoding(transformer_dim, drop=drop_cpe)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            dim_feedforward=feedforward_dim,
            nhead=num_heads,
            dropout=drop_transformer,
        )
        self.classifier = nn.Linear(feature_dim + projection_dim, num_classes)

    def _apply_transformer(self, image_feats: torch.Tensor, lens):
        B, N, C, H, W = image_feats.shape
        image_feats = image_feats.flatten(start_dim=2).permute([1, 0, 2])  # [N, B, C * H * W]
        image_feats = self.pos_encoding(image_feats, lens)
        image_feats = self.transformer(image_feats)
        return image_feats.permute([1, 0, 2]).reshape([B, N, C, H, W])

    def _pool(self, image_feats, lens):
        if self.pooling == "last_timestep":
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, int(l) - 1])
        elif self.pooling == "sum":
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, : int(l)].sum(0))
        else:
            raise ValueError(f"Unkown pooling method: {self.pooling}")

        pooled_feats = torch.stack(pooled_feats)
        pooled_feats = F.adaptive_avg_pool2d(pooled_feats, (1, 1))
        return pooled_feats.squeeze(3).squeeze(2)

    def forward(self, images, lens):
        B, N, C, H, W = images.shape
        images = images.reshape([B * N, C, H, W])
        # Apply Image Model
        image_feats = self.image_model(images)
        image_feats = F.relu(self.group_norm(image_feats))
        # Apply transformer
        image_feats_proj = self.projection(image_feats).reshape(
            [B, N, -1, *self.image_shape]
        )
        image_feats_trans = self._apply_transformer(image_feats_proj, lens)
        # Concat and apply classifier
        image_feats = image_feats.reshape([B, N, -1, *self.image_shape])
        image_feats_combined = torch.cat([image_feats, image_feats_trans], dim=2)
        image_feats_pooled = self._pool(image_feats_combined, lens)
        return self.classifier(image_feats_pooled)
