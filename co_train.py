import os
import glob
import argparse
import json
from urllib import parse
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn import svm, ensemble
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

# for augmentation, see https://github.com/hassony2/torch_videovision
from torchvideotransforms import video_transforms, volume_transforms

from dataset import CGDataset, ISDataset, ParkinsonsDiseaseSubset
from transform import RandomDropSlice, Interpolate, Padding
from co_model import VGG, VolumeModel, Classifier, VGGwithEmbedding, VGGwithAttn, VGGwithMultiAttn
from utils import infinite_loader, abbreviate, to_device

parser = argparse.ArgumentParser()
# settings
parser.add_argument("--dir_name", type=str)
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--val_per_step", type=int, default=30)
parser.add_argument("--test", action='store_true', default=False)
# transform
parser.add_argument("--num_slices", type=int, default=32)
# models
parser.add_argument("--co_layers", type=int, default=16)
parser.add_argument("--total_layers", type=int, default=29)
parser.add_argument("--image_shape", default=(4, 4))
parser.add_argument("--hidden_dim", type=int, default=128)
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
    # RandomDropSlice(),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    # Padding(args.num_slices),
    Interpolate(args.num_slices),
]
train_transform_list_IS = [
    video_transforms.RandomRotation(5),
    video_transforms.CenterCrop((50, 50)),
    video_transforms.Resize((72, 72)),
    # RandomDropSlice(),
    volume_transforms.ClipToTensor(channel_nb=3, div_255=False),
    # Padding(args.num_slices),
    Interpolate(args.num_slices),
]
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
train_transform_CG = video_transforms.Compose(train_transform_list_CG)
valid_transform_CG = video_transforms.Compose(valid_transform_list_CG)
train_transform_IS = video_transforms.Compose(train_transform_list_IS)
valid_transform_IS = video_transforms.Compose(valid_transform_list_IS)


def propagate(loader, model, classifier, split, hospital):
    output_list = []
    pred_list = []
    label_list = []
    feature_list = []
    info_list = []
    with torch.no_grad():
        for image, size, age, gender, label in loader:
            image, size, age, gender = to_device(
                [image, size, age, gender], device)

            output = model(image, size, return_attn=True)
            output, features = classifier(output, age, gender, size)
            pred = torch.argmax(output.data, axis=1)
            output_list.append(output.cpu())
            pred_list.append(pred.cpu())
            label_list.append(label.cpu())
            feature_list.append([f.cpu() for f in features])
            info_list.append(torch.stack([age, gender], dim=1).cpu())

    output = torch.cat(output_list, dim=0)
    pred = torch.cat(pred_list, dim=0)
    label = torch.cat(label_list, dim=0)
    feature = [torch.cat(f, dim=0) for f in zip(*feature_list)]
    info = torch.cat(info_list, dim=0)

    loss = nn.functional.cross_entropy(output, label)
    acc = (pred.numpy() == label.numpy()).mean()
    return {
        '%s_acc_%s_DL' % (split, hospital): acc,
        '%s_loss_%s_DL' % (split, hospital): loss}, pred, label, feature, info


def valid(loaders, model, classifier, hospital,
          CLFs=[svm.SVC, partial(ensemble.RandomForestClassifier, n_jobs=-1)]):
    metrics = dict()
    pred_splits = dict()
    label_splits = dict()
    feature_splits = dict()
    info_splits = dict()
    for split, loader in loaders.items():
        results, pred, label, feature, info = propagate(
            loader, model, classifier, split, hospital)
        metrics.update(results)
        pred_splits['%s_%s_DL' % (split, hospital)] = pred.numpy()
        label_splits['%s_%s' % (split, hospital)] = label.numpy()
        feature_splits[split] = feature
        info_splits[split] = info
        
    for CLF in CLFs:
        clf = CLF()
        clf.fit(
            torch.cat([feature_splits['train'][0], info_splits['train']], dim=1).numpy(),
            label_splits['train_%s' % hospital])
        for split in loaders.keys():
            preds = clf.predict(
                torch.cat([feature_splits[split][0], info_splits[split]], dim=1).numpy())
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
        print('CG:', train_idx_CG.shape, valid_idx_CG.shape, test_idx_CG.shape)
        print('IS:', train_idx_IS.shape, valid_idx_IS.shape, test_idx_IS.shape)

        # class weight
        class_num_CG = [
            Counter(labels_CG[train_idx_CG])[i] for i in range(n_class_CG)]
        class_num_IS = [
            Counter(labels_IS[train_idx_IS])[i] for i in range(n_class_IS)]
        class_weight_CG = len(train_idx_CG) / torch.FloatTensor(class_num_CG)
        class_weight_IS = len(train_idx_IS) / torch.FloatTensor(class_num_IS)
        class_weight_CG = class_weight_CG / class_weight_CG.sum()
        class_weight_IS = class_weight_IS / class_weight_IS.sum()
        weight_CG = [class_weight_CG[i] for i in labels_CG[train_idx_CG]]
        weight_IS = [class_weight_IS[i] for i in labels_IS[train_idx_IS]]
        weight_CG = torch.FloatTensor(weight_CG)
        weight_IS = torch.FloatTensor(weight_IS)        

        # --------------- split ---------------
        # train
        train_set_CG = ParkinsonsDiseaseSubset(dataset_CG, train_idx_CG, train_transform_CG)
        train_set_IS = ParkinsonsDiseaseSubset(dataset_IS, train_idx_IS, train_transform_IS)
        train_loader_CG = DataLoader(
            train_set_CG, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        train_loader_IS = DataLoader(
            train_set_IS, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        infinite_CG_loader = infinite_loader(DataLoader(
            train_set_CG, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True, drop_last=True
        ))
        infinite_IS_loader = infinite_loader(DataLoader(
            train_set_IS, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True, drop_last=True
        ))
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
        max_acc = defaultdict(float)
        backend = models.vgg16(pretrained=True, progress=True)
        # backend.features[30] = nn.Identity()
        backend.avgpool = nn.AdaptiveAvgPool2d(1)  # 1 or 4
        backend.classifier = nn.Identity()
        for param in backend.parameters():
            param.requires_grad = True
        model = VGGwithMultiAttn(backend).to(device)
        classifier_CG = Classifier(n_class_CG, args.hidden_dim).to(device)
        classifier_IS = Classifier(n_class_IS, args.hidden_dim).to(device)

        params = (
            list(model.parameters()) +
            list(classifier_CG.parameters()) +
            list(classifier_IS.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.alpha)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, total_steps=args.total_steps,
            max_lr=args.lr, anneal_strategy='cos'
        )

        # 記得改class weight
        criterion_CG = nn.CrossEntropyLoss(weight=class_weight_CG).to(device)
        criterion_IS = nn.CrossEntropyLoss(weight=class_weight_IS).to(device)

        metrics_list = defaultdict(list)
        with tqdm(range(args.total_steps), dynamic_ncols=True) as progress:
            for step in progress:
                # train
                model.train()
                classifier_CG.train()
                classifier_IS.train()

                image_CG, size_CG, age_CG, gender_CG, label_CG = next(infinite_CG_loader)
                image_CG, size_CG, age_CG, gender_CG, label_CG = to_device(
                    [image_CG, size_CG, age_CG, gender_CG, label_CG], device)

                image_IS, size_IS, age_IS, gender_IS, label_IS = next(infinite_IS_loader)
                image_IS, size_IS, age_IS, gender_IS, label_IS = to_device(
                    [image_IS, size_IS, age_IS, gender_IS, label_IS], device)

                optimizer.zero_grad()
                feature_CG = model.forward_CG(image_CG, size_CG)
                feature_IS = model.forward_IS(image_IS, size_IS)
                output_CG, _ = classifier_CG(feature_CG, age_CG, gender_CG, size_CG)
                output_IS, _ = classifier_IS(feature_IS, age_IS, gender_IS, size_IS)

                loss_CG = criterion_CG(output_CG, label_CG)
                loss_IS = criterion_IS(output_IS, label_IS)
                loss = loss_CG + loss_IS
                loss.backward()
                optimizer.step()
                scheduler.step()

                # valid
                model.eval()
                classifier_CG.eval()
                classifier_IS.eval()

                if step % args.val_per_step == 0:
                    metrics_CG, preds_CG, trues_CG = valid(
                        {'train': train_loader_CG,
                         'valid': valid_loader_CG,
                         'test': test_loader_CG},
                        model.forward_CG, classifier_CG, 'CG')
                    metrics_IS, preds_IS, trues_IS = valid(
                        {'train': train_loader_IS,
                         'valid': valid_loader_IS,
                         'test': test_loader_IS},
                        model.forward_IS, classifier_IS, 'IS')
                    
                    msg = []
                    best_msg = []
                    metrics = {**metrics_CG, **metrics_IS}
                    preds = {**preds_CG, **preds_IS}
                    trues = {**trues_CG, **trues_IS}
                    for name, value in metrics.items():
                        metrics_list[name].append(value)
                        if name.startswith('valid_acc'):
                            _, _, hospital, model_name = name.split('_')
                            if value > max_acc[name]:
                                max_acc[name] = value
                                torch.save({
                                    'model': model.state_dict(),
                                    'classifier_CG': classifier_CG.state_dict(),
                                    'classifier_IS': classifier_IS.state_dict(),
                                    'max_acc': metrics['test_acc_%s_%s' % (hospital, model_name)],
                                    'pred': preds['test_%s_%s' % (hospital, model_name)],
                                    'label': trues['test_%s' % hospital]
                                }, args.dir_name + '/fold_%d_best_%s_%s.pt' % (
                                    fold + 1, abbreviate(model_name), hospital))
                            msg.append('%s_%s: %.3f' % (abbreviate(model_name), hospital, value))
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
            plt.figure(figsize=(15, 5))
            for idx, hospital in enumerate(['CG', 'IS']):
                plt.subplot(1, 2, idx + 1)
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
            f1_macro = f1_score(ckpt['label'], ckpt['pred'], average='macro', zero_division=0)
            results['%s_%s_accs' % (hospital[:-3], model_name)].append(ckpt['max_acc'])
            results['%s_%s_macro' % (hospital[:-3], model_name)].append(f1_macro)

    result_CG = defaultdict(list)
    result_IS = defaultdict(list)
    for name, values in results.items():
        hospital, model_name, score = name.split('_')
        values = np.array(values)
        msg = '%.4f (%.2f)' % (values.mean(), values.std())
        if hospital == 'CG':
            result_CG['%s_avg' % model_name].append(msg)
        else:
            result_IS['%s_avg' % model_name].append(msg)
    
    df_CG = pd.DataFrame.from_dict(result_CG, orient='index', columns=['acc', 'f_macro'])
    df_IS = pd.DataFrame.from_dict(result_IS, orient='index', columns=['acc', 'f_macro'])

    print('CG \n', df_CG.to_latex(), '\n')
    print('IS \n', df_IS.to_latex(), '\n')        

if __name__ == "__main__":
    if args.test:
        test()
    else:
        train()
