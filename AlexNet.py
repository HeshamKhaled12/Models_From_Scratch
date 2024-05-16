
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


class Alexnet(torch.nn.Module):
  def __init__(self,num_classes):
    super().__init__()
    self.features=nn.Sequential(
    nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2,bias=False),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
    nn.Conv2d(96,192,kernel_size=5 ,stride=1,padding=2,bias=False),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
    nn.Conv2d(192,384,kernel_size=3,stride=1,padding=1,bias=False),
    nn.ReLU(inplace=True),
    nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1,bias=False),
    nn.ReLU(inplace=True),
    nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=0))

    self.classifier=nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096,4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096,num_classes)
    )

  def forward(self,x):
    x=self.features(x)
    x=x.view(x.size(0),256*6*6)
    x=self.classifier(x)
    return x
