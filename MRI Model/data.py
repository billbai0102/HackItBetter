import torch
from torch.utils.data import Dataset, DataLoader

import cv2

class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Initializes dataset class
        :param df: DataFrame containing MRI and segmentation mask paths
        :param transform: Image transforms
        """
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        mri = cv2.imread(self.df.iloc[idx]['MRI'])
        mask = cv2.imread(self.df.iloc[idx]['Mask'], 0)

        if self.transform is not None:
            mri = self.transform(mri)
            mask = self.transform(mask)

        return {
            'MRI': mri,
            'Mask': mask
        }

    def __len__(self):
        return len(self.df)
