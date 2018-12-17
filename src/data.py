import torch
from torch.utils.data import Dataset
import h5py
from PIL import Image
import numpy as np

class FacialDataset(Dataset):
    def __init__(self, h5_filename, img_height, img_width):
        super(FacialDataset, self).__init__()

        hf = h5py.File(h5_filename, 'r')
        self.images = hf['images']
        self.labels = hf['labels']
        
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index].reshape([self.img_height, self.img_width])
        img_pil = Image.fromarray(image.astype(np.uint8), 'L')
        degree = np.random.randint(-15,16)
        img_pil = img_pil.rotate(degree)
        image = np.array(img_pil).reshape(-1,).astype(np.float32)

        image = torch.FloatTensor(image).view(1, self.img_height, self.img_width)
        image = image / 255.0
        label = self.labels[index]
        return image, label
