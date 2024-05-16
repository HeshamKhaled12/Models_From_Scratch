
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


class VGG16 (nn.Module):
  def __init__ (self,num_classes):
    super().__init__()

    self.block1=nn.Sequential(
        nn.Conv2d(3,64,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    )


    self.block2=nn.Sequential(
        nn.Conv2d(64,128,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    )


    self.block3=nn.Sequential(
        nn.Conv2d(128,256,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    )


    self.block4=nn.Sequential(
        nn.Conv2d(256,512,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    )

    self.block5=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    )

    self.classifier=nn.Sequential(
        nn.Linear(512*7*7,4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,num_classes)
    )

    for m in self.modules():
      if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight,mode='fan_in',nonlinearity='relu')
        if m.bias is not None:
          m.bias.detach().zero_()


  def forward(self, x):
    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=self.block4(x)
    x=self.block5(x)
    x=x.view(x.size(0),-1)
    x=self.classifier(x)
    return x


