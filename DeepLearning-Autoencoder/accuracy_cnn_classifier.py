#I WROTE THIS




import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.cnn import CNN
from train_common import *
from utils import config
import utils


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

def evaluate_cnn(dataset, model):

    num_classes = config('cnn.num_classes')
    batch_size = config('cnn.batch_size')
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
            print("accuracy is for class ", c, " ", (correct/total) * 100)
def main():
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('cnn.num_classes'))
    dataset = get_data_by_label(va_loader)


    model = CNN()

    print('Loading cnn...')
    model, start_epoch, stats = restore_checkpoint(model, config('cnn.checkpoint'))


    # Evaluate model
    evaluate_cnn(dataset, model)

    # Report performance
    #report_validation_performance(dataset, get_semantic_label, model, criterion)

if __name__ == '__main__':
    main()
