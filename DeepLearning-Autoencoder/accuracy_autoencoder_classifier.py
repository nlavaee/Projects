"""
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Evaluate Autoencoder
    Runs inference on an autoencoder to get the per-class performance on the
    validation data.
    Usage: python evaluate_autoencoder.py
"""

# I WROTE THIS
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
from dataset import get_train_val_test_loaders
from model.autoencoder import AutoencoderClassifier
from train_common import *
from utils import config

def get_data_by_label(dataset):
    data = []
    for i, (X, y) in enumerate(dataset):
        for c in range(config('autoencoder.classifier.num_classes')):
            if len((y == c).nonzero()) != 0:
                batch = X[(y == c).nonzero().squeeze(1)]
                if len(data) <= c:
                    data.append(batch)
                else:
                    data[c] = torch.cat((data[c], batch))
    return data

def evaluate_autoencoder(dataset, model):
    num_classes = config('autoencoder.classifier.num_classes')
    batch_size = config('autoencoder.classifier.batch_size')
    correct = 0
    total = 0
    performance = np.zeros(num_classes)
    with torch.no_grad():
        for c in range(num_classes):
            correct = 0
            X = dataset[c]
            output = model(X)
            label = c
            predicted = predictions(output.data)
            total = X.size(0)
            correct += (predicted == label).sum().item()
            print("accuracy is for class " , c, " ",  (correct/total) * 100)

def main():
    # data loaders
    _, va_loader, _, get_semantic_label = get_train_val_test_loaders(
        num_classes=config('autoencoder.classifier.num_classes'))
    dataset = get_data_by_label(va_loader)

    model = AutoencoderClassifier(config('autoencoder.ae_repr_dim'),
        config('autoencoder.classifier.num_classes'))
    criterion = torch.nn.MSELoss()

    # Attempts to restore the latest checkpoint if exists
    print('Loading autoencoder...')
    model, start_epoch, _ = restore_checkpoint(model,
        config('autoencoder.classifier.checkpoint'))

    # Evaluate model
    evaluate_autoencoder(dataset, model)

    # Report performance
    #report_validation_performance(dataset, get_semantic_label, model, criterion)

if __name__ == '__main__':
    main()