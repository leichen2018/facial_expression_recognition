import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import cnn
from visualization.gradcam import GradCam
from visualization.misc_functions import save_class_activation_images
import h5py
from PIL import Image
import random
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='FER CNN visualization')

parser.add_argument('--test_data', default='../data/public_test.h5', type=str)

parser.add_argument('--img_height', default=48, type=int)
parser.add_argument('--img_width', default=48, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--model_type', default='fer_resnet18', type=str)
parser.add_argument('--model_path', default='../checkpoint/model_0.pth', type=str)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--idx', default=0, type=int)

args = parser.parse_args()

if args.model_type == 'vgg11':
    model = cnn.vgg11()
elif args.model_type == 'vgg11_bn':
    model = cnn.vgg11_bn()
if args.model_type == 'vgg13':
    model = cnn.vgg13()
elif args.model_type == 'vgg13_bn':
    model = cnn.vgg13_bn()
elif args.model_type == 'vgg16':
    model = cnn.vgg16()
elif args.model_type == 'fer_vgg13':
    model = cnn.fer_vgg13_bn()
elif args.model_type == 'fer_resnet18':
    model = cnn.fer_resnet18()
elif args.model_type == 'fer_resnet34':
    model = cnn.fer_resnet34()
elif args.model_type == 'fer_resnet50':
    model = cnn.fer_resnet50()
elif args.model_type == 'fer_resnet101':
    model = cnn.fer_resnet101()
elif args.model_type == 'fer_resnet152':
    model = cnn.fer_resnet152()
print(model)

model.load_state_dict(torch.load(args.model_path))

device = torch.device('cuda') if args.cuda else torch.device('cpu')
model.to(device)

label_dict = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'sad',
    'surprise',
    'neutral'
]

hf = h5py.File(args.test_data)
images = hf['images']
labels = hf['labels']
idx = args.idx
#idx = random.randint(0, len(images) - 1)
print(idx)
ori_img = images[idx].reshape((48, 48))
prep_img = torch.from_numpy(ori_img).float().to(device).view(1, 1, 48, 48)
prep_img /= 255.0

finalconv_name = 'layer4'
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.cpu().detach().numpy())

model._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].cpu().detach().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (48, 48)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

logit = model(prep_img)

h_x = F.softmax(logit, dim=1).detach().squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().detach().numpy()
print(idx)
idx = idx.cpu().detach().numpy()

print('{:.3f} -> {}'.format(probs[0], label_dict[idx[0]]))

for i in range(7):
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[i]])
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (48, 48)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + np.stack((ori_img,)*3, axis=-1) * 0.5
    cv2.imwrite(str(i)+'_'+label_dict[idx[i]]+'.jpg', result)
