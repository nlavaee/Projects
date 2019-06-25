"""
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Dogs Dataset
    Class wrapper for interfacing with the dataset of dog images
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
from torch.utils.data import Dataset, DataLoader
from utils import config

def get_train_val_test_loaders(num_classes):
    tr, va, te, _ = get_train_val_dataset(num_classes=num_classes)
    
    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_dataset(num_classes=10):
    tr = DogsDataset('train', num_classes)
    va = DogsDataset('val', num_classes)
    te = DogsDataset('test', num_classes)
        
    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)
   
    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)
    
    return tr, va, te, standardizer


class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1. 
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.
    
    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None
    
    def fit(self, X):
        # TODO: Complete this function
        # size of X is (7665, 32, 32, 3) ~ so this means that there are 7,

        self.image_mean = []
        self.image_std = []

        # print(np.mean(X, axis=tuple(range(X.ndim-1))))
        # print(np.std(X, axis=tuple(range(X.ndim-1))))

        for i in range(X.shape[-1]):
            self.image_mean.append(X[:, :, :, i].mean())
            self.image_std.append(X[:, :, :, i].std())
        
    
    def transform(self, X):
        # TODO: Complete this function
        #print(X)
        self.image_mean = np.asarray(self.image_mean)
        X = X.astype(float)

        for i in range(X.shape[-1]):
            X[:, :, :, i] -= self.image_mean[i]
            X[:, :, :, i] /= self.image_std[i]

        return X
      

class DogsDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()
        
        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))
        
        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes
        
        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'), index_col=0)
        self.X, self.y = self._load_data()
    
        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()
    
    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % self.partition)
        
        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            elif self.num_classes == 10:
                df = self.metadata[self.metadata.partition.isin([self.partition, ' '])]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5 or 10')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]
        
        X, y = [], []
        for i, row in df.iterrows():
            label = row['numeric_label']
            image = imread(os.path.join(config('image_path'), row['filename']))
            X.append(image)
            y.append(row['numeric_label'])
            if self.partition == 'train' or self.partition == 'validate':
                X.append(np.flip(image, 0))
                y.append(row['numeric_label'])


        
        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("Image Mean:", standardizer.image_mean)
    print("Image Std: ", standardizer.image_std)
