import csv
import numpy as np
import os
import h5py

"""
Please unzip fer2013.tar before running the script.
"""

if __name__ == '__main__':
    images = []
    labels = []
    splits = []

    print('Reading files...')
    with open('fer2013/fer2013.csv') as csvfile:
        next(csvfile, None)
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            images.append(row[1])
            labels.append(row[0])
            splits.append(row[2])

    image_np = []
    for image in images:
        image = np.array([int(p) for p in image.split(' ')], dtype=np.float32)
        image_np.append(image)
    image_np = np.stack(image_np)

    label_np = []
    for label in labels:
        label = np.array(int(label))
        label_np.append(label)
    label_np = np.stack(label_np)

    print('Preparing H5DF data...')
    train = {}
    train['images'] = []
    train['labels'] = []
    public_test = {}
    public_test['images'] = []
    public_test['labels'] = []
    private_test = {}
    private_test['images'] = []
    private_test['labels'] = []
    for image, label, split in zip(images, labels, splits):
        if split == 'Training':
            train['images'].append([int(p) for p in image.split(' ')])
            train['labels'].append(int(label))
        elif split == 'PublicTest':
            public_test['images'].append([int(p) for p in image.split(' ')])
            public_test['labels'].append(int(label))
        elif split == 'PrivateTest':
            private_test['images'].append([int(p) for p in image.split(' ')])
            private_test['labels'].append(int(label))
        else:
            raise ValueError('Wrong Split %s' % split)

    print('Saving H5DF data...')
    training_filename = 'train.h5'
    with h5py.File(training_filename, 'w') as hf:
        hf.create_dataset('images', data=train['images'])
        hf.create_dataset('labels', data=train['labels'])
    public_test_filename = 'public_test.h5'
    with h5py.File(public_test_filename, 'w') as hf:
        hf.create_dataset('images', data=public_test['images'])
        hf.create_dataset('labels', data=public_test['labels'])
    private_test_filename = 'private_test.h5'
    with h5py.File(private_test_filename, 'w') as hf:
        hf.create_dataset('images', data=private_test['images'])
        hf.create_dataset('labels', data=private_test['labels'])
    
    print('Done!')