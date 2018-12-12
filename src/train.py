import torch
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description='Facial Expression Recognition Kaggle Contest')
parser.add_argument('--train_data', default='../data/train.h5', type=str)
parser.add_argument('--test_data', default='../data/public_test.h5', type=str)
parser.add_argument('--model_type', default='SVM', type=str)

args = parser.parse_args()

if args.model_type.lower() == 'svm':
    from sklearn import svm
    with h5py.File(args.train_data, 'r') as hf:
        images = np.array(hf['images'], dtype=np.float32)
        labels = np.array(hf['labels'])
    images = images / 255.0
    model = svm.SVC(gamma='scale')

    print('Fitting SVM...')
    model.fit(images, labels)

    print('Predicting Testing Samples...')
    with h5py.File(args.test_data, 'r') as hf:
        test_images = np.array(hf['images'], dtype=np.float32)
        test_labels = np.array(hf['labels'])
    test_images = test_images / 255.0
    pred = model.predict(test_images)
    acc = np.equal(pred, test_labels).sum()
    print('Acc: %.3f (%d/%d)' % (acc / test_images.shape[0], acc, test_images.shape[0]))
    