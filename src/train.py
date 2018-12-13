import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import h5py
import numpy as np
from data import FacialDataset
from cnn import LeNet

parser = argparse.ArgumentParser(description='Facial Expression Recognition Kaggle Contest')

parser.add_argument('--train_data', default='../data/train.h5', type=str)
parser.add_argument('--test_data', default='../data/public_test.h5', type=str)

parser.add_argument('--img_height', default=48, type=int)
parser.add_argument('--img_width', default=48, type=int)
parser.add_argument('--batch_size', default=64, type=int)

parser.add_argument('--model_type', default='SVM', type=str)

parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--print_every', default=10, type=int)

args = parser.parse_args()

def train(batch_idx, model, dataloader, device, optimizer):
    model.train()
    
    for i, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()

        acc = torch.eq(output.argmax(dim=1), label).sum().item()
        if i % args.print_every == 0:
            print('Epoch[%d] [%d/%d] Loss: %.4f Acc: %d/%d' %
                (batch_idx, i, len(dataloader), loss.item(), acc, img.size(0)))

def test(batch_idx, model, dataloader, device):
    model.eval()

    with torch.no_grad():
        acc = 0
        cnt = 0
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)

            output = model(img)

            acc += torch.eq(output.argmax(dim=1), label).sum().item()
            cnt += img.size(0)
    
    print('Test...Epoch[%d] Acc: %.2f (%d/%d)' % 
        (batch_idx, acc / cnt, acc, cnt))


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
elif args.model_type.lower() == 'cnn':
    train_dataset = FacialDataset(args.train_data, args.img_height, args.img_width)
    test_dataset = FacialDataset(args.test_data, args.img_height, args.img_width)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, \
            batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, \
            batch_size=args.batch_size, shuffle=False)
    
    model = LeNet()
    print(model)

    if args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.to(device)

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(),\
                lr=args.lr, momentum=args.momentum, nesterov=True)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)


    for i in range(args.epochs):
        train(i, model, train_dataloader, device, optimizer)
        test(i, model, test_dataloader, device)