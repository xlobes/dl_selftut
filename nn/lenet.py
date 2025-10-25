import torch
import torch.nn as nn
import pdb


class LeNet5(nn.Module):
    """
    Classic LeNet-5 for grayscale digits.
    Expects input shaped (N, 1, 32, 32). If you pass (N, 1, 28, 28) (MNIST),
    it will zero-pad by 2 pixels on each side to make it 32x32 internally.
    """
    def __init__(self, num_classes = 10):
        super().__init__()

        # define layer i
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # define layer ii
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # define layer iii
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.ReLU()

        # define layer v: fully connected
        self.fc4 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()

        # define layer iv: fully connected for output
        self.fc_out = nn.Linear(84, num_classes)

    def forward(self, x_img):

        # apply layer i
        x1 = self.conv1(x_img)     # when you run this line: stop and write x1.shape; then compare it with x.shape
        x1 = self.act1(x1)
        x1 = self.pool1(x1)
        # print(f'layer i: {x1.shape}')

        # apply layer ii
        x2 = self.conv2(x1)
        x2 = self.act2(x2)
        x2 = self.pool2(x2)
        # print(f'layer ii: {x2.shape}')

        # apply layer iii
        x3 = self.conv3(x2)
        x3 = self.act3(x3)
        # print(f'layer iii: {x3.shape}')

        # apply layer iv
        x4 = torch.flatten(x3, start_dim=1) # open up the third layers and go from matrix to vector

        x4 = self.fc4(x4)
        x4 = self.act4(x4)
        # print(f'layer iv: {x4.shape}')

        # apply layer v
        logits = self.fc_out(x4)
        return logits


if __name__ == "__main__":
    x = torch.randn(1, 1, 28, 28)         # x is an image
    f_net = LeNet5(num_classes=10)        # f is a function which is a neural network
    y_predict = f_net.forward(x)
 