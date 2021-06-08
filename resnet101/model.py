from jittor.models import Resnet101
import jittor.nn as nn 

class Net(nn.Module):
    def __init__(self, num_classes):
        self.base_net = Resnet101(num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        return x 