import os
import glob
import argparse
import json
from posix import listdir
from urllib import parse
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn import svm, ensemble
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models

# for augmentation, see https://github.com/hassony2/torch_videovision
from torchvideotransforms import video_transforms, volume_transforms

from dataset import CGDataset, ISDataset, ParkinsonsDiseaseSubset
from transform import RandomDropSlice, Interpolate, Padding
from co_model import VGG, VolumeModel, Classifier, VGGwithEmbedding, VGGwithAttn, VGGwithMultiAttn
from utils import infinite_loader

parser = argparse.ArgumentParser()
# settings
parser.add_argument("--dir_name", type=str)
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=1)
# transform
parser.add_argument("--num_slices", type=int, default=32)
# learning
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()


# device
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
if torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device, args.device)

valid_transform_list_CG = [
    video_transforms.CenterCrop((72, 72)),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    # Padding(args.num_slices),
    Interpolate(args.num_slices),
]
valid_transform_list_IS = [
    video_transforms.CenterCrop((50, 50)),
    video_transforms.Resize((72, 72)),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    # Padding(args.num_slices),
    Interpolate(args.num_slices),
]
valid_transform_CG = video_transforms.Compose(valid_transform_list_CG)
valid_transform_IS = video_transforms.Compose(valid_transform_list_IS)


def valid(loaders, model):
    w_splits = dict()
    for split, loader in loaders.items():
        w_list = []
        with torch.no_grad():
            for image, size, _, _, _ in tqdm(loader):
                image, size = image.to(device), size.to(device)
                _, w = model(image, size, return_attn=True)
                w_list.append(w.cpu())
        w = torch.cat(w_list, dim=0)
        w_splits[split] = w
    return w_splits


def plot():
    dataset_CG = CGDataset()
    dataset_IS = ISDataset()

    labels_CG = dataset_CG.labels
    n_class_CG = len(Counter(labels_CG))
    print('(CG) number of classes:', n_class_CG, Counter(labels_CG))
    labels_IS = dataset_IS.labels
    n_class_IS = len(Counter(labels_IS))
    print('(IS) number of classes:', n_class_IS, Counter(labels_IS))

    # kfolds
    kfolds = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold in range(args.folds):
        # train index, valid_index, test_index
        data_idx_CG, test_idx_CG = list(
            kfolds.split(np.arange(len(labels_CG)), labels_CG))[fold]
        train_idx_CG, valid_idx_CG = train_test_split(
            data_idx_CG, test_size=0.2, random_state=args.seed,
            stratify=labels_CG[data_idx_CG])
        data_idx_IS, test_idx_IS = list(
            kfolds.split(np.arange(len(labels_IS)), labels_IS))[fold]
        train_idx_IS, valid_idx_IS = train_test_split(
            data_idx_IS, test_size=0.2, random_state=args.seed,
            stratify=labels_IS[data_idx_IS])
        print('fold %d:' % (fold + 1))

        # --------------- split ---------------
        # train
        train_set_CG = ParkinsonsDiseaseSubset(dataset_CG, train_idx_CG, valid_transform_CG)
        train_set_IS = ParkinsonsDiseaseSubset(dataset_IS, train_idx_IS, valid_transform_IS)
        train_loader_CG = DataLoader(
            train_set_CG, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        train_loader_IS = DataLoader(
            train_set_IS, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        # valid
        valid_set_CG = ParkinsonsDiseaseSubset(dataset_CG, valid_idx_CG, valid_transform_CG)
        valid_set_IS = ParkinsonsDiseaseSubset(dataset_IS, valid_idx_IS, valid_transform_IS)
        valid_loader_CG = DataLoader(
            valid_set_CG, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        valid_loader_IS = DataLoader(
            valid_set_IS, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        # test
        test_set_CG = ParkinsonsDiseaseSubset(dataset_CG, test_idx_CG, valid_transform_CG)
        test_set_IS = ParkinsonsDiseaseSubset(dataset_IS, test_idx_IS, valid_transform_IS)
        test_loader_CG = DataLoader(
            test_set_CG, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader_IS = DataLoader(
            test_set_IS, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        # settings
        backend = models.vgg16(pretrained=True, progress=True)
        # backend.features[30] = nn.Identity()
        backend.avgpool = nn.AdaptiveAvgPool2d(1)
        backend.classifier = nn.Identity()
        model = VGGwithAttn(backend).to(device)
        
        ckpt = torch.load(os.path.join(args.dir_name, 'fold_%d_best_DL_CG.pt' % (fold + 1)))
        model.load_state_dict(ckpt['model'])
        model.eval()
        w_splits = valid(
            {'train': train_loader_CG,
             'valid': valid_loader_CG,
             'test': test_loader_CG},
            model.forward_CG)
        width = 0.3
        for i, (name, w) in enumerate(w_splits.items()):
            w = w.mean(dim=0).numpy()
            x = np.arange(args.num_slices) + i * width
            plt.bar(x, w, edgecolor='white', width=width, label=name)
        plt.xticks(np.arange(args.num_slices) + width, np.arange(args.num_slices))
        plt.legend()
        plt.savefig(args.dir_name + '/CG_fold_%d.png' % (fold + 1))
        plt.clf()

        ckpt = torch.load(os.path.join(args.dir_name, 'fold_%d_best_DL_IS.pt' % (fold + 1)))
        model.load_state_dict(ckpt['model'])
        model.eval()
        w_splits = valid(
            {'train': train_loader_IS,
             'valid': valid_loader_IS,
             'test': test_loader_IS},
            model.forward_IS)
        width = 0.3
        for i, (name, w) in enumerate(w_splits.items()):
            w = w.mean(dim=0).numpy()
            x = np.arange(args.num_slices) + i * width
            plt.bar(x, w, edgecolor='white', width=width, label=name)
        plt.xticks(np.arange(args.num_slices) + width, np.arange(args.num_slices))
        plt.legend()
        plt.savefig(args.dir_name + '/IS_fold_%d.png' % (fold + 1))
        plt.clf()

if __name__ == "__main__":
    plot()
