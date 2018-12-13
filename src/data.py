import torch
from torch.utils.data import Dataset
import h5py

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
        image = torch.FloatTensor(self.images[index]).view(1, self.img_height, self.img_width)
        image = image / 255.0
        label = self.labels[index]
        return image, label