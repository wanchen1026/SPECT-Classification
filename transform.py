import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class RandomDropSlice:
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, slices):
        ndrop = int(len(slices) * self.ratio)
        head, tail = torch.randint(0, ndrop, (2,))
        return slices[head:len(slices) - tail]


class Padding:
    def __init__(self, channel):
        self.channel = channel

    def __call__(self, slices):
        slices = torch.transpose(slices, 0, 1)
        if self.channel < len(slices):
            left = (len(slices) - self.channel) // 2
            right = len(slices) - self.channel - left
            slices = slices[left:len(slices) - right]
        size = len(slices)
        if self.channel == size:
            return slices, size
        else:
            pad = []
            for _ in range(self.channel - len(slices)):
                pad.append(torch.zeros_like(slices[0]))
            pad = torch.stack(pad, dim=0)
            slices = torch.cat((slices, pad), dim=0)
            return slices, size


class Interpolate:
    def __init__(self, channel=32):
        self.channel = channel

    def __call__(self, slices):
        slices = slices.unsqueeze(dim=0)
        slices = F.interpolate(
            slices, mode='trilinear',
            align_corners=False,
            size=(self.channel, slices.shape[-2], slices.shape[-1])
        )
        return torch.transpose(slices[0], 0, 1), self.channel
