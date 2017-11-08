import torch
import torch.onnx
import torchvision
from agnet import AGNet
from torch.autograd import Variable

net = AGNet()
net.eval()
x = Variable(torch.randn(2, 3, 224, 224))
torch.onnx.export(net, x, './model/agnet.proto', verbose=True)

x = Variable(torch.ones(2,3,224,224))
print(net(x))
