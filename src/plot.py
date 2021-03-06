import h5py
import numpy as np
import cv2
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description='Facial Expression Recognition Feature Extraction')

parser.add_argument('--pca', action="store_true")
args = parser.parse_args()

maps = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
image_maps = [0, 299, 2, 7, 3, 15, 4]
use_pca = args.pca

if args.pca:
    directory = 'svm_pca/' 
    model_name = 'svm_pca'
else:
    directory = 'svm/'
    model_name = 'svm'

with h5py.File('../data/train.h5', 'r') as hf:
    images_origin = np.array(hf['images'], dtype=np.float32)
    labels = np.array(hf['labels'])
   
images_origin = normalize(images_origin)
images_origin = images_origin
labels = labels

model = svm.LinearSVC()

if use_pca:
    print('Use PCA...')
    n_components = 128
    pca = PCA(n_components)
    print(f'n_components={n_components}')
    pca.fit(images_origin)
    print(f'PCA Before: {images_origin.shape}')
    images = pca.transform(images_origin)
    print(f'PCA After: {images.shape}')
else:
    images = images_origin

print('Fitting SVM...')
model.fit(images, labels)

train_pred = model.predict(images)

train_acc = np.equal(train_pred, labels).sum()
print('Train Acc: %.3f (%d/%d)' % (train_acc / images.shape[0], train_acc, images.shape[0]))

print('Predicting Testing Samples...')
with h5py.File('../data/public_test.h5', 'r') as hf:
    test_images_origin = np.array(hf['images'], dtype=np.float32)
    test_labels = np.array(hf['labels'])


test_images_n = normalize(test_images_origin)

test_images_n = test_images_n
test_labels = test_labels

if use_pca:
    test_images = pca.transform(test_images_n)
else:
    test_images = test_images_n
pred = model.predict(test_images)

acc = np.equal(pred, test_labels).sum()
print('Test Acc: %.3f (%d/%d)' % (acc / test_images.shape[0], acc, test_images.shape[0]))

if use_pca:
    b = pca.inverse_transform(model.coef_)
else:
    b = model.coef_

print(b.shape)

for i in range(test_images.shape[0]):
    I = b[test_labels[i]].copy()
    I = I * test_images_n[i]
    I = (I - np.min(I)) / (np.max(I) - np.min(I)) # normalization
    I *= 255.0
    I = I.reshape([48,48]).astype(np.uint8)
    ori = test_images_origin[i].reshape([48,48]).astype(np.uint8)
    #ori = (ori * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(I, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + np.stack((ori,)*3, axis=-1) * 0.5
    cv2.imwrite(directory + str(i) + '_' + maps[test_labels[i]] + '.png',result)
    if pred[i] != test_labels[i]:
        j = pred[i]
        I = b[j].copy()
        I = I * test_images_n[i]
        I = (I - np.min(I)) / (np.max(I) - np.min(I)) # normalization
        I *= 255.0
        I = I.reshape([48,48]).astype(np.uint8)
        ori = test_images_origin[i].reshape([48,48]).astype(np.uint8)
        #ori = (ori * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(I, cv2.COLORMAP_JET)
        result = heatmap * 0.3 + np.stack((ori,)*3, axis=-1) * 0.5
        cv2.imwrite(directory + str(i) + '_wrong_predict_' + maps[j] + '.png',result)


print('Saving model...')
pickle.dump(model, open('model/'+model_name, 'wb'))
print('Done...')


# for i in range(b.shape[0]):
#     print()
#     I = b[i]
#     I = (I - np.min(I)) / (np.max(I) - np.min(I)) # normalization
#     I *= 255.0
#     I = I.reshape([48,48]).astype(np.uint8)
#     heatmap = cv2.applyColorMap(I, cv2.COLORMAP_JET)
#     ori = images_origin[image_maps[i]].reshape([48,48])
#     ori = (ori * 255.0).astype(np.uint8)
#     result = heatmap * 0.3 + np.stack((ori,)*3, axis=-1) * 0.5
#     cv2.imwrite(directory + str(i) + '_' + maps[i] + '.png',result)
