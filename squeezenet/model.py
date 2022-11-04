import torch
import torch.nn as nn
import math

# reference: https://blog.csdn.net/fupotui7870/article/details/115732660
class QuadraticPolynomialActivation(nn.Module):
    def __init__(self, ):
        super(QuadraticPolynomialActivation, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        self.weight.data.uniform_(-0.5, 0.5)
    def forward(self, input):
        return self.weight[0] * input**2 + self.weight[1] * input + self.weight[2]

# reference 1 (HEMET): https://arxiv.org/pdf/2106.00038.pdf
# reference 2 (EVA): https://arxiv.org/pdf/1912.11951.pdf
# reference 3: https://github.com/kaizouman/tensorsandbox/tree/master/cifar10/models/squeeze
class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1, bias=False)  
        self.qpa1 = QuadraticPolynomialActivation()
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1, bias=False)
        self.qpa2 = QuadraticPolynomialActivation()
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.qpa3 = QuadraticPolynomialActivation()
        self.bn3 = nn.BatchNorm2d(expand_planes)     
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
    def forward(self, x):
        x = self.conv1(x)
        x = self.qpa1(x)
        x = self.bn1(x)
        out1 = self.conv2(x)
        out1 = self.qpa2(out1)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.qpa3(out2)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        return out

                                    # slots     # 4096            # 8192 less     # 8192          # 16384
CONV1_FILTER_NUM =                  3           # 3               # 7             # 7             # 14
FIRE2_SQUEEZE_PLAINES =             12          # 12              # 20            # 25            # 32
FIRE2_EXPAND_PLAINES =              8           # 8               # 16            # 16            # 32
FIRE3_SQUEEZE_PLAINES =             12          # 12              # 20            # 25            # 32
FIRE3_EXPAND_PLAINES =              6           # 6               # 12            # 12            # 25
CONV2_FILTER_NUM =                  40          # 40              # 20            # 81            # 160
CONV3_FILTER_NUM =                  40          # 40              # 20            # 81            # 160



class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, CONV1_FILTER_NUM, kernel_size=3, stride=1, padding=1, bias=False)   
        self.qpa1 = QuadraticPolynomialActivation()
        self.bn1 = nn.BatchNorm2d(CONV1_FILTER_NUM)
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire2 = fire(CONV1_FILTER_NUM, FIRE2_SQUEEZE_PLAINES, FIRE2_EXPAND_PLAINES)
        self.fire3 = fire(FIRE2_EXPAND_PLAINES * 2, FIRE3_SQUEEZE_PLAINES, FIRE3_EXPAND_PLAINES)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)   
        
        self.conv2 = nn.Conv2d(FIRE3_EXPAND_PLAINES * 2, CONV2_FILTER_NUM, kernel_size=3, stride=1, padding=1, bias=False)    
        self.qpa2 = QuadraticPolynomialActivation()
        self.bn2 = nn.BatchNorm2d(CONV2_FILTER_NUM)
        
        self.conv3 = nn.Conv2d(CONV2_FILTER_NUM, CONV3_FILTER_NUM, kernel_size=3, stride=1, padding=1, bias=False)
        self.qpa3 = QuadraticPolynomialActivation()
        self.bn3 = nn.BatchNorm2d(CONV3_FILTER_NUM)
        
        self.conv4 = nn.Conv2d(CONV3_FILTER_NUM, 10, kernel_size=1, stride=1, bias=False)
        self.qpa4 = QuadraticPolynomialActivation()
        self.bn4 = nn.BatchNorm2d(10)

        self.avgpool3 = nn.AvgPool2d(kernel_size=8, stride=8)   

        self.softmax = nn.LogSoftmax(dim=1)      

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        x = self.conv1(x)
        x = self.qpa1(x)
        x = self.bn1(x)     
        x = self.avgpool1(x)

        x = self.fire2(x)

        x = self.fire3(x)

        x = self.avgpool2(x)

        x = self.conv2(x)
        x = self.qpa2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.qpa3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.qpa4(x)
        x = self.bn4(x)

        x = self.avgpool3(x)

        x = self.softmax(x)

        return x
