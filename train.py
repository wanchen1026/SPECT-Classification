import os
import glob
import argparse
import json
from urllib import parse
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import append
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

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
from utils import infinite_loader, abbreviate, to_device
from transform import Padding, Interpolate, SlicesToTensors
from model import (
    VGGplusLinear, VGGplusConv2d, VGGwithACS, VolumeModel,
    VGGwithEmbedding, VGGwithAttn, VGGwithMultiAttn)

parser = argparse.ArgumentParser()
# settings
parser.add_argument("--hospital", type=str, default='CG')
parser.add_argument("--dir_name", type=str)
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--run_only_these_folds", type=int, nargs='+', default=[0, 1, 2, 3, 4])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--val_per_step", type=int, default=30)
parser.add_argument("--test", action='store_true', default=False)
# transform
parser.add_argument("--num_slices", type=int, default=32)
# learning
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--total_steps", type=int, default=3000)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()


# device
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
if torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device, args.device)

# augmentation
train_transform_list_CG = [
    video_transforms.RandomRotation(5),
    video_transforms.CenterCrop((72, 72)),
    video_transforms.Resize((72, 72)),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    Interpolate(args.num_slices),
]
train_transform_list_IS = [
    video_transforms.RandomRotation(5),
    video_transforms.CenterCrop((50, 50)),
    video_transforms.Resize((72, 72)),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    Interpolate(args.num_slices),
]
valid_transform_list_CG = [
    video_transforms.CenterCrop((72, 72)),
    video_transforms.Resize((72, 72)),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    Interpolate(args.num_slices),
]
valid_transform_list_IS = [
    video_transforms.CenterCrop((50, 50)),
    video_transforms.Resize((72, 72)),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    Interpolate(args.num_slices),
]

if args.hospital == 'CG':
    train_transform = video_transforms.Compose(train_transform_list_CG)
    valid_transform = video_transforms.Compose(valid_transform_list_CG)
else:
    train_transform = video_transforms.Compose(train_transform_list_IS)
    valid_transform = video_transforms.Compose(valid_transform_list_IS)


def propagate(loader, model, split, hospital):
    output_list = []
    pred_list = []
    label_list = []
    feature_list = []
    info_list = []
    with torch.no_grad():
        for image, size, age, gender, label in loader:
            image, size, age, gender = to_device([image, size, age, gender], device)
            output, feature = model(image, age, gender, size)
            pred = torch.argmax(output.data, axis=1)
            output_list.append(output.cpu())
            pred_list.append(pred.cpu())
            label_list.append(label.cpu())
            feature_list.append(feature.cpu())
            info_list.append(torch.stack([age, gender], dim=1).cpu())

    output = torch.cat(output_list, dim=0)
    pred = torch.cat(pred_list, dim=0)
    label = torch.cat(label_list, dim=0)
    feature = torch.cat(feature_list, dim=0)
    info = torch.cat(info_list, dim=0)

    loss = nn.functional.cross_entropy(output, label)
    acc = (pred.numpy() == label.numpy()).mean()
    return {
        '%s_acc_%s_DL' % (split, hospital): acc,
        '%s_loss_%s_DL' % (split, hospital): loss}, pred, label, feature, info


def valid(loaders, model, hospital,
          CLFs=[svm.SVC, partial(ensemble.RandomForestClassifier, n_jobs=-1)]):
    metrics = dict()
    pred_splits = dict()
    label_splits = dict()
    feature_splits = dict()
    info_splits = dict()
    for split, loader in loaders.items():
        results, pred, label, feature, info = propagate(
            loader, model, split, hospital)
        metrics.update(results)
        pred_splits['%s_%s_DL' % (split, hospital)] = pred.numpy()
        label_splits['%s_%s' % (split, hospital)] = label.numpy()
        feature_splits[split] = feature
        info_splits[split] = info

    for CLF in CLFs:
        clf = CLF()
        clf.fit(
            torch.cat([feature_splits['train'], info_splits['train']], dim=1).numpy(),
            label_splits['train_%s' % hospital])
        for split in loaders.keys():
            preds = clf.predict(
                torch.cat([feature_splits[split], info_splits[split]], dim=1).numpy())
            trues = label_splits['%s_%s' % (split, hospital)]
            pred_splits['%s_%s_%s' % (split, hospital, clf.__class__.__name__)] = preds
            acc = (preds == trues).mean()
            metrics['%s_acc_%s_%s' % (split, hospital, clf.__class__.__name__)] = acc

    return metrics, pred_splits, label_splits


def train():
    os.makedirs(args.dir_name, exist_ok=False)
    print(json.dumps(vars(args), indent=4))
    json.dump(
        vars(args), 
        open(os.path.join(args.dir_name, 'args.json'), 'w'),
        indent=4)

    dataset = CGDataset() if args.hospital == 'CG' else ISDataset()
    labels = dataset.labels
    n_class = len(Counter(labels))
    print(args.hospital, 'number of classes:', n_class, Counter(labels))

    # kfolds
    kfolds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold in range(args.folds):
        if fold not in args.run_only_these_folds:
            continue
        # train index, valid_index, test_index
        data_idx, test_idx = list(kfolds.split(np.arange(len(labels)), labels))[fold]
        train_idx, valid_idx = train_test_split(
            data_idx, test_size=0.2, random_state=args.seed, stratify=labels[data_idx])
        print('fold %d:' % (fold + 1))
        print('train: %d, valid: %d, test: %d' % (
            train_idx.shape[0], valid_idx.shape[0], test_idx.shape[0]))

        # class weight
        class_num = [Counter(labels[train_idx])[i] for i in range(n_class)]
        class_weight = len(train_idx) / torch.FloatTensor(class_num)
        class_weight = class_weight / class_weight.sum()
        weight = [class_weight[i] for i in labels[train_idx]]
        weight = torch.FloatTensor(weight)

        # --------------- split ---------------
        # train
        train_set_inf = ParkinsonsDiseaseSubset(dataset, train_idx, train_transform)
        infinite_train_loader = infinite_loader(DataLoader(
            train_set_inf, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True, drop_last=True
        ))
        train_set = ParkinsonsDiseaseSubset(dataset, train_idx, valid_transform)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False)
        # valid
        valid_set = ParkinsonsDiseaseSubset(dataset, valid_idx, valid_transform)
        valid_loader = DataLoader(
            valid_set, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False)
        # test
        test_set = ParkinsonsDiseaseSubset(dataset, test_idx, valid_transform)
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False)

        # settings
        max_acc = defaultdict(float)
        backend = models.vgg16(pretrained=True, progress=True)
        backend.features[30] = nn.Identity()
        backend.avgpool = nn.AdaptiveAvgPool2d(1) # 1 or 4
        backend.classifier = nn.Identity()
        for param in backend.parameters():
            param.requires_grad = True
        model = VGGwithAttn(backend, n_class).to(device) # models are in model.py

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, total_steps=args.total_steps,
            max_lr=args.lr, anneal_strategy='cos'
        )
        criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)

        metrics_list = defaultdict(list)
        with tqdm(range(args.total_steps), dynamic_ncols=True) as progress:
            for step in progress:
                # train
                model.train()
                image, size, age, gender, label = next(infinite_train_loader)
                image, size, age, gender, label = to_device(
                    [image, size, age, gender, label], device)
                output, _ = model(image, age, gender, size)
                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # valid
                model.eval()
                if step % args.val_per_step == 0:
                    metrics, preds, trues = valid(
                        {'train': train_loader,
                         'valid': valid_loader,
                         'test': test_loader},
                        model, args.hospital)
                    
                    msg = []
                    best_msg = []
                    metrics = {**metrics}
                    preds = {**preds}
                    trues = {**trues}
                    for name, value in metrics.items():
                        metrics_list[name].append(value)
                        if name.startswith('valid_acc'):
                            _, _, hospital, model_name = name.split('_')
                            if value > max_acc[name]:
                                max_acc[name] = value
                                torch.save({
                                    'model': model.state_dict(),
                                    'max_acc': metrics['test_acc_%s_%s' % (hospital, model_name)],
                                    'pred': preds['test_%s_%s' % (hospital, model_name)],
                                    'label': trues['test_%s' % hospital]
                                }, args.dir_name + '/fold_%d_best_%s_%s.pt' % (
                                    fold + 1, abbreviate(model_name), hospital))
                            msg.append('%s_%s: %.3f' % (
                                abbreviate(model_name), hospital, value))
                            best_msg.append('best_%s_%s: %.3f' % (
                                abbreviate(model_name), hospital, max_acc[name]))
                    progress.write(
                        ('step: %d, %s, lr: %.6f') % (
                            step,
                            ','.join(msg),
                            optimizer.param_groups[0]['lr']))
                    progress.set_postfix_str('%s' % (' '.join(best_msg)))

        model_names = set()
        for name in metrics_list.keys():
            if name.startswith('valid_acc'):
                _, _, _, model_name = name.split('_')
                model_names.add(model_name)
        model_names = list(model_names)

        # plot (accuracy)
        for model_name in model_names:
            x = np.arange(0, args.total_steps, args.val_per_step)
            plt.figure(figsize=(7.5, 5))
            for idx, hospital in enumerate([args.hospital]):
                for name, value in metrics_list.items():
                    if name.endswith('_acc_%s_%s' % (hospital, model_name)):
                        split = name.split('_')[0]
                        plt.plot(x, value, label=split)
                plt.xlabel('Number of Steps')
                plt.ylabel('Accuracy')
                plt.title('%s %s Acc' % (hospital, model_name))
                plt.legend(loc='upper right')
            plt.savefig(args.dir_name + '/fold_%d_acc_%s.png' % (
                fold + 1, abbreviate(model_name)))


def test():
    args_dict = json.load(open(os.path.join(args.dir_name, 'args.json')))
    args.__dict__.update(args_dict)

    results = defaultdict(list)
    for fold in tqdm(range(args.folds)):
        paths = glob.glob(args.dir_name + '/fold_%d_best_*.pt' % (fold + 1))

        for path in paths:
            ckpt = torch.load(path)
            file_name = os.path.basename(path)
            _, _, _, model_name, hospital = file_name.split('_')
            f1 = f1_score(ckpt['label'], ckpt['pred'], average='macro', zero_division=0)
            results['%s_%s_accs' % (hospital[:-3], model_name)].append(ckpt['max_acc'])
            results['%s_%s_f1' % (hospital[:-3], model_name)].append(f1)

    avg = defaultdict(list)
    for name, values in results.items():
        hospital, model_name, score = name.split('_')
        values = np.array(values)
        avg['%s_avg' % model_name].append('%.4f (%.2f)' % (values.mean(), values.std()))
    df = pd.DataFrame.from_dict(avg, orient='index')
    print(args.hospital, '\n', df.to_latex(), '\n')

if __name__ == "__main__":
    if args.test:
        test()
    else:
        train()
