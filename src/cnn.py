import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self, img_height=48, img_width=48):
        super(LeNet, self).__init__()

        # 48 x 48
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 44 x 44
        self.pool1 = nn.MaxPool2d(2)
        # 22 x 22
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 18 x 18
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2)
        # 9 x 9
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(20 * 9 * 9, 50)
        self.fc2 = nn.Linear(50, 7)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 20 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)