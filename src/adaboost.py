import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import h5py
import numpy as np
from data import FacialDataset
from cnn import LeNet
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='Facial Expression Recognition Kaggle Contest')

parser.add_argument('--train_data', default='../data/train.h5', type=str)
parser.add_argument('--test_data', default='../data/public_test.h5', type=str)

parser.add_argument('--img_height', default=48, type=int)
parser.add_argument('--img_width', default=48, type=int)
parser.add_argument('--batch_size', default=64, type=int)

parser.add_argument('--model_type', default='Adaboost', type=str)

parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--print_every', default=10, type=int)

parser.add_argument('--pca', action="store_true")
parser.add_argument('--kernel', default='rbf', type=str)
parser.add_argument('--kernelpca', action="store_true")
parser.add_argument('--pca_n', default=5, type=int)

parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--algo', type=str, default='SAMME')

args = parser.parse_args()

if args.model_type.lower() == 'adaboost':
    print(args.model_type)
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.tree import DecisionTreeClassifier
    print('learing rate: '+str(args.lr))
    print('max_depth: '+str(args.depth))
    print('n_estimators: '+str(args.n_estimators))
    print('algo: '+args.algo)
    with h5py.File(args.train_data, 'r') as hf:
        images = np.array(hf['images'], dtype=np.float32)
        labels = np.array(hf['labels'])
    images = images / 255.0
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=args.depth), n_estimators=args.n_estimators, learning_rate=args.lr, algorithm=args.algo)

    if args.pca or args.kernelpca:
        if args.pca:
            print('Use PCA...')
            pca = PCA(n_components='mle')
        if args.kernelpca:
            print('Use Kenerl PCA...')
            pca = KernelPCA(n_components=args.pca_n, kernel=args.kernel)
        pca.fit(images)
        print(f'PCA Before: {images.shape}')
        images = pca.transform(images)
        print(f'PCA After: {images.shape}')

    print('Fitting '+args.model_type+'...')
    model.fit(images, labels)

    print('Predicting Testing Samples...')
    with h5py.File(args.test_data, 'r') as hf:
        test_images = np.array(hf['images'], dtype=np.float32)
        test_labels = np.array(hf['labels'])
    test_images = test_images / 255.0
    if args.pca or args.kernelpca:
        test_images = pca.transform(test_images)
    pred = model.predict(test_images)
   
    acc = np.equal(pred, test_labels).sum()
    print('Acc: %.3f (%d/%d)' % (acc / test_images.shape[0], acc, test_images.shape[0]))
