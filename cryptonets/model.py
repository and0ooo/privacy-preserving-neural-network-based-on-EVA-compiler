import torch
import torch.nn as nn
import math



# reference: https://blog.csdn.net/fupotui7870/article/details/115732660
class QuadraticPolynomialActivation(nn.Module):

    def __init__(self, ):
        super(QuadraticPolynomialActivation, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(3),
                                         requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-0.5, 0.5)

    def forward(self, input):
        return self.weight[0] * input**2 + self.weight[
            1] * input + self.weight[2]


# reference: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf
class CryptoNets(nn.Module):

    def __init__(self):
        super(CryptoNets, self).__init__()
        self.conv1 = nn.Conv2d(1,
                               3,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               bias=False)
        self.qpa1 = QuadraticPolynomialActivation()
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3,
                               5,
                               kernel_size=5,
                               stride=2,
                               padding=0,
                               bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(5 * 5 * 5, 10, bias=False)
        self.softmax = nn.LogSoftmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.qpa1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        batch_size, _, _, _ = x.shape
        x = x.view(batch_size, 5 * 5 * 5)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
