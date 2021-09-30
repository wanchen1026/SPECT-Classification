import os
import pydicom
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset


class ParkinsonsDataset(Dataset):
    def __init__(self, paths, ages, genders, labels, starts, cutpoint, select=False):
        self.ages = ages
        self.genders = genders
        self.labels = labels
        self.starts = starts
        self.images = []
        for idx, image_path in enumerate(paths):
            image = pydicom.dcmread(image_path).pixel_array
            image = torch.from_numpy(image / image.max()).float()
            if select:
                slices = image[int(starts[idx]) + 3:int(starts[idx]) + 6]
                self.images.append(slices)    
            else:
                slices = []
                for i in range(image.shape[0]):
                    if (image[i] > 0.1).sum() > cutpoint:
                        slices.append(image[i])
                self.images.append(torch.stack(slices))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        age = torch.tensor(self.ages[idx] / 100, dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.float32)
        label = self.labels[idx]
        slices = self.images[idx]
        return slices, age, gender, label


class CGDataset(ParkinsonsDataset):
    def __init__(self):
        image_loc = '../data/ChangGung/TRODAT_SPECT_DICOM/'
        data = pd.read_excel('../data/ChangGung/TRODAT_SPECT_New(n=634).xlsx')
        IDs = np.array(data['研究編號'])

        ages = np.array(data['年齡'])
        genders = np.array(data['性別\n(0=女，1=男)'])
        labels = np.array(data['判讀結果(四分法)'])
        starts = np.array(data['影像起始張數\n(分析連續9張)'])

        paths = []
        for ID in IDs:
            image_path = os.path.join(image_loc, '%03d.IMA' % ID)
            paths.append(image_path)

        super().__init__(paths, ages, genders, labels, starts, 800, select=True)


class ISDataset(ParkinsonsDataset):
    def __init__(self):
        image_loc = '../data/E-DA/DICOMS'
        slices = pd.read_csv('../data/E-DA/Selected_Slice_Index.csv')
        data = pd.read_excel('../data/E-DA/NCTU_PD_Data_list_2017_202.xlsx')
        IDs = np.array(data['ID'])

        ages = np.array(data['Age'])
        genders = np.array(data['性別'])
        genders = [1 if gender == 'M' else 0 for gender in genders]
        labels = np.array(data[' 分期'])
        labels[labels == 'N'] = 0
        labels = labels.astype('int64')

        starts = []
        paths = []
        for ID in IDs:
            start = slices.loc[slices['FileName'] == ID, 'SliceIndex']
            starts.append(int(start) - 4)
            file_path = os.path.join(image_loc, ID)
            file_names = next(os.walk(file_path))[2]
            file_name = (
                file_names[1]
                if file_names[0].startswith('._') else file_names[0]
            )
            image_path = os.path.join(file_path, file_name)
            paths.append(image_path)

        super().__init__(paths, ages, genders, labels, starts, 400, select=True)


class ParkinsonsDiseaseSubset(Subset):

    def __init__(self, dataset, indices, transform):
        self.transform = transform
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        slices, age, gender, label = super().__getitem__(idx)
        slices = list(map(transforms.functional.to_pil_image, slices))
        slices, size = self.transform(slices)
        return slices, size, age, gender, label
